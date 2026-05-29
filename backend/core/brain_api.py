"""
BRAIN API wrapper using q3yi/worldquant SDK.
Handles authentication, session management, polling, and error recovery.
"""
import time
import logging
from typing import Dict, Optional, List, Tuple
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from backend.core.expression_normalizer import normalize_brain_expression
from backend.core.simulation_settings import DEFAULT_SIMULATION_SETTINGS as BASE_SIMULATION_SETTINGS
from backend.utils.time import utc_now

logger = logging.getLogger(__name__)


class BRAINRateLimitError(RuntimeError):
    """Raised when BRAIN asks the client to slow down."""

    def __init__(self, message: str = "BRAIN API rate limit exceeded", retry_after: Optional[float] = None):
        super().__init__(message)
        self.retry_after = retry_after


class BRAINAuthenticationError(RuntimeError):
    """Raised when BRAIN rejects credentials or an expired session."""


class BRAINSession:
    """
    Manages a persistent WorldQuant BRAIN API session.
    Handles auth, rate limiting, cookie recovery, and polling.
    """
    
    BASE_URL = "https://api.worldquantbrain.com"
    DEFAULT_SIMULATION_SETTINGS = BASE_SIMULATION_SETTINGS
    
    def __init__(self, email: str, password: str, session_name: str = "default"):
        self.email = email
        self.password = password
        self.session_name = session_name
        self.session = requests.Session()
        self.session.auth = (email, password)
        self.is_authenticated = False
        self.auth_timestamp = None
        self.rate_limit_remaining = None
        self.rate_limit_reset = None
        self.last_status_code = None
        self.last_error = None
        self.last_retry_after = None
        
        # Setup retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
    
    def authenticate(self) -> bool:
        """
        Authenticate with BRAIN API using email/password.
        Returns True on success, False otherwise.
        """
        try:
            response = self.session.post(
                f"{self.BASE_URL}/authentication",
                timeout=10
            )
            self.last_status_code = response.status_code
            self.last_retry_after = self._retry_after(response)
            
            if response.status_code in [200, 201]:
                self.is_authenticated = True
                self.auth_timestamp = utc_now()
                self.last_error = None
                logger.info(f"[{self.session_name}] Authenticated successfully: {self.email}")
                return True
            else:
                self.last_error = response.text
                logger.error(f"[{self.session_name}] Auth failed: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"[{self.session_name}] Auth error: {e}")
            return False
    
    def _update_rate_limits(self, response: requests.Response):
        """Extract rate limit headers from response."""
        if "x-ratelimit-remaining" in response.headers:
            self.rate_limit_remaining = int(response.headers["x-ratelimit-remaining"])
        if "x-ratelimit-reset" in response.headers:
            self.rate_limit_reset = int(response.headers["x-ratelimit-reset"])
    
    @staticmethod
    def _retry_after(response: requests.Response) -> Optional[float]:
        value = response.headers.get("Retry-After")
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _handle_rate_limit(self, response: requests.Response) -> bool:
        """
        Handle 429 (Too Many Requests) without blocking the dashboard request.
        Raises BRAINRateLimitError so callers can retry later.
        """
        if response.status_code == 429:
            retry_after = self._retry_after(response)
            self.last_status_code = 429
            self.last_retry_after = retry_after
            self.last_error = response.text or "BRAIN API rate limit exceeded"
            logger.warning(f"[{self.session_name}] Rate limited. Retry after {retry_after or 'unknown'}s")
            raise BRAINRateLimitError("BRAIN API rate limit exceeded; retry polling later", retry_after)
        return False

    def _reauthenticate_once(self) -> bool:
        """Refresh cookies after BRAIN rejects an otherwise-authenticated request."""
        logger.warning(f"[{self.session_name}] BRAIN returned 401; refreshing authentication session")
        try:
            self.session.close()
        except Exception:
            pass
        self.session = requests.Session()
        self.session.auth = (self.email, self.password)
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        self.is_authenticated = False
        return self.authenticate()

    def _authentication_error(self, response: requests.Response) -> BRAINAuthenticationError:
        self.last_status_code = response.status_code
        self.last_error = response.text
        return BRAINAuthenticationError(
            f"BRAIN authentication failed during {self.session_name}: {response.text or response.status_code}"
        )
    
    def submit_expression(
        self,
        expression: str,
        universe: str = "default",
        settings: Optional[Dict] = None,
    ) -> Optional[str]:
        """
        Submit an alpha expression to BRAIN simulator.
        Returns simulation_id on success, None otherwise.
        
        POST /simulations
        Body follows BRAIN simulation schema:
        { "type": "REGULAR", "settings": {...}, "regular": expression }
        Response Location header contains a polling URL.
        """
        if not self.is_authenticated:
            logger.error(f"[{self.session_name}] Not authenticated, cannot submit")
            return None
        
        try:
            simulation_settings = self.DEFAULT_SIMULATION_SETTINGS.copy()
            if settings:
                simulation_settings.update(settings)
            if universe and universe != "default":
                simulation_settings["universe"] = universe

            payload = {
                "type": "REGULAR",
                "settings": simulation_settings,
                "regular": normalize_brain_expression(expression),
            }
            
            response = self.session.post(f"{self.BASE_URL}/simulations", json=payload, timeout=15)
            if response.status_code == 401 and self._reauthenticate_once():
                response = self.session.post(f"{self.BASE_URL}/simulations", json=payload, timeout=15)
            
            self._update_rate_limits(response)
            
            if self._handle_rate_limit(response):
                return self.submit_expression(expression, universe, settings=settings)  # Retry
            
            if response.status_code in [201, 200]:
                progress_url = response.headers.get("Location", "")
                if progress_url:
                    logger.info(f"[{self.session_name}] Submitted expression: {progress_url}")
                    return progress_url

            if response.status_code == 401:
                raise self._authentication_error(response)
            
            logger.error(f"[{self.session_name}] Submit failed: {response.status_code} - {response.text}")
            return None
        except BRAINRateLimitError:
            raise
        except BRAINAuthenticationError:
            raise
        except Exception as e:
            logger.error(f"[{self.session_name}] Submit error: {e}")
            return None
    
    def get_simulation_status(self, simulation_id: str) -> Optional[Dict]:
        """
        Get simulation status and progress.
        
        GET /simulations/{simulation_id}
        Response: { "progress": 0-100, "status": "...", ... }
        """
        if not self.is_authenticated:
            return None
        
        try:
            url = simulation_id if simulation_id.startswith("http") else f"{self.BASE_URL}/simulations/{simulation_id}"
            response = self.session.get(url, timeout=10)
            if response.status_code == 401 and self._reauthenticate_once():
                response = self.session.get(url, timeout=10)
            
            self._update_rate_limits(response)
            
            if self._handle_rate_limit(response):
                return self.get_simulation_status(simulation_id)
            
            if response.status_code == 200:
                retry_after = float(response.headers.get("Retry-After", 0) or 0)
                if retry_after > 0:
                    return {"status": "running", "progress": 0, "retry_after": retry_after}
                payload = response.json()
                if payload.get("alpha"):
                    payload.setdefault("status", "completed")
                    payload.setdefault("progress", 100)
                return payload

            if response.status_code == 401:
                raise self._authentication_error(response)
            
            logger.error(f"[{self.session_name}] Get status failed: {response.status_code}")
            return None
        except BRAINRateLimitError:
            raise
        except BRAINAuthenticationError:
            raise
        except Exception as e:
            logger.error(f"[{self.session_name}] Get status error: {e}")
            return None
    
    def get_alpha_results(self, simulation_id: str) -> Optional[Dict]:
        """
        Get full alpha backtest results (only after simulation completes).
        
        GET /alphas/{alpha_id} or GET /simulations/{simulation_id}/alpha
        Response: { "sharpe": 1.5, "fitness": 1.2, "turnover": 25, ... }
        """
        if not self.is_authenticated:
            return None
        
        try:
            alpha_id = None
            if simulation_id.startswith("http"):
                progress_response = self.session.get(simulation_id, timeout=10)
                if progress_response.status_code == 401 and self._reauthenticate_once():
                    progress_response = self.session.get(simulation_id, timeout=10)
                if self._handle_rate_limit(progress_response):
                    return self.get_alpha_results(simulation_id)
                if progress_response.status_code == 200 and float(progress_response.headers.get("Retry-After", 0) or 0) == 0:
                    alpha_id = progress_response.json().get("alpha")

            response = self.session.get(
                f"{self.BASE_URL}/alphas/{alpha_id or simulation_id}",
                timeout=10
            )
            if response.status_code == 401 and self._reauthenticate_once():
                response = self.session.get(
                    f"{self.BASE_URL}/alphas/{alpha_id or simulation_id}",
                    timeout=10,
                )
            
            self._update_rate_limits(response)
            
            if self._handle_rate_limit(response):
                return self.get_alpha_results(simulation_id)
            
            if response.status_code == 200:
                return response.json()

            if response.status_code == 401:
                raise self._authentication_error(response)

            if not simulation_id.startswith("http"):
                fallback = self.session.get(
                    f"{self.BASE_URL}/simulations/{simulation_id}/alpha",
                    timeout=10,
                )
                if fallback.status_code == 401 and self._reauthenticate_once():
                    fallback = self.session.get(
                        f"{self.BASE_URL}/simulations/{simulation_id}/alpha",
                        timeout=10,
                    )
                self._update_rate_limits(fallback)
                if self._handle_rate_limit(fallback):
                    return self.get_alpha_results(simulation_id)
                if fallback.status_code == 200:
                    return fallback.json()
                if fallback.status_code == 401:
                    raise self._authentication_error(fallback)
            
            logger.warning(f"[{self.session_name}] Get results returned {response.status_code}")
            return None
        except BRAINRateLimitError:
            raise
        except BRAINAuthenticationError:
            raise
        except Exception as e:
            logger.error(f"[{self.session_name}] Get results error: {e}")
            return None
    
    def poll_until_complete(self, simulation_id: str, max_wait: int = 3600, poll_interval: int = 10) -> Tuple[bool, Optional[Dict]]:
        """
        Poll simulation until completion or timeout.
        Returns (completed, results).
        """
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            status = self.get_simulation_status(simulation_id)
            
            if not status:
                logger.error(f"[{self.session_name}] Failed to get status for {simulation_id}")
                time.sleep(poll_interval)
                continue
            
            progress = status.get("progress", 0)
            logger.info(f"[{self.session_name}] Simulation {simulation_id}: {progress}%")
            
            # Check if complete
            if status.get("status") == "completed" or progress == 100:
                results = self.get_alpha_results(simulation_id)
                return (True, results)
            
            time.sleep(poll_interval)
        
        logger.error(f"[{self.session_name}] Simulation {simulation_id} timed out after {max_wait}s")
        return (False, None)
    
    def get_data_fields(
        self,
        dataset_id: Optional[str] = None,
        search: Optional[str] = None,
        region: str = "USA",
        universe: str = "TOP3000",
        delay: int = 1,
        limit: int = 1000,
    ) -> Optional[List[Dict]]:
        """
        Fetch available BRAIN data fields.
        Returns list of field definitions.
        """
        if not self.is_authenticated:
            return None

        params = {
            "instrumentType": "EQUITY",
            "region": region,
            "universe": universe,
            "delay": delay,
            "limit": limit,
        }
        if dataset_id:
            params["dataset.id"] = dataset_id
        if search:
            params["search"] = search

        try:
            response = self._get_with_endpoint_fallback(
                endpoints=("data-fields", "data_fields"),
                params=params,
                timeout=30,
            )

            if response.status_code == 200:
                fields = self._items_from_payload(response.json())
                logger.info(f"[{self.session_name}] Fetched {len(fields)} data fields")
                return fields

            logger.error(f"[{self.session_name}] Fetch fields failed: {response.status_code}")
            return None
        except BRAINRateLimitError:
            raise
        except Exception as e:
            logger.error(f"[{self.session_name}] Fetch fields error: {e}")
            return None

    def get_datasets(
        self,
        region: str = "USA",
        universe: str = "TOP3000",
        delay: int = 1,
    ) -> Optional[List[Dict]]:
        """Fetch BRAIN dataset metadata when available from the live API."""
        if not self.is_authenticated:
            return None

        params = {
            "instrumentType": "EQUITY",
            "region": region,
            "universe": universe,
            "delay": delay,
        }
        try:
            response = self._get_with_endpoint_fallback(
                endpoints=("data-sets", "data_sets", "datasets"),
                params=params,
                timeout=30,
            )
            if response.status_code == 200:
                return self._items_from_payload(response.json())
            logger.error(f"[{self.session_name}] Fetch datasets failed: {response.status_code}")
            return None
        except BRAINRateLimitError:
            raise
        except Exception as e:
            logger.error(f"[{self.session_name}] Fetch datasets error: {e}")
            return None

    def _get_with_endpoint_fallback(
        self,
        endpoints: Tuple[str, ...],
        params: Optional[Dict] = None,
        timeout: int = 30,
    ) -> requests.Response:
        response = None
        for endpoint in endpoints:
            response = self.session.get(f"{self.BASE_URL}/{endpoint}", params=params, timeout=timeout)
            self.last_status_code = response.status_code
            self.last_retry_after = self._retry_after(response)
            self._handle_rate_limit(response)
            if response.status_code not in {404, 405}:
                return response
        return response

    @staticmethod
    def _items_from_payload(payload) -> List[Dict]:
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            for key in ("results", "data", "items"):
                value = payload.get(key)
                if isinstance(value, list):
                    return value
        return []
    
    def close(self):
        """Close session."""
        self.session.close()
        self.is_authenticated = False


class BRAINClient:
    """
    High-level client for BRAIN API operations.
    Wraps BRAINSession with convenience methods.
    """
    
    def __init__(self, email: str, password: str):
        self.session = BRAINSession(email, password)
        if not self.session.authenticate():
            raise RuntimeError("Failed to authenticate with BRAIN API")
    
    def submit_and_wait(self, expression: str, timeout: int = 3600) -> Optional[Dict]:
        """
        Submit expression and wait for results.
        Returns backtest results dict or None on failure.
        """
        sim_id = self.session.submit_expression(expression)
        if not sim_id:
            return None
        
        completed, results = self.session.poll_until_complete(sim_id, max_wait=timeout)
        return results if completed else None
    
    def close(self):
        self.session.close()


# Convenience function for testing
def test_brain_connection(email: str, password: str) -> bool:
    """Test BRAIN API connection."""
    try:
        client = BRAINClient(email, password)
        client.close()
        return True
    except Exception as e:
        logger.error(f"BRAIN connection test failed: {e}")
        return False
