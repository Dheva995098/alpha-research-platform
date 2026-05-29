"""
Test suite for BRAIN API integration.
"""
import pytest
from backend.core.brain_api import BRAINSession, BRAINClient
from backend.core.data_fields import BRAINDataFields, get_data_fields
from backend.security import encrypt_credential, decrypt_credential, generate_aes_key


class TestSecurity:
    """Test credential encryption/decryption."""
    
    def test_encrypt_decrypt(self):
        """Test basic encryption/decryption."""
        plaintext = "my-secret-password"
        encrypted = encrypt_credential(plaintext)
        decrypted = decrypt_credential(encrypted)
        assert decrypted == plaintext
    
    def test_aes_key_generation(self):
        """Test AES key generation."""
        key = generate_aes_key()
        assert key is not None
        assert len(key) > 0
        assert isinstance(key, str)


class TestDataFields:
    """Test data fields schema and validation."""
    
    def test_initialization(self):
        """Test schema initialization."""
        schema = BRAINDataFields()
        assert len(schema.fields) > 0
        assert "close" in schema.fields
        assert "volume" in schema.fields
    
    def test_validate_field(self):
        """Test field validation."""
        schema = BRAINDataFields()
        assert schema.validate_field("close") is True
        assert schema.validate_field("nonexistent_field") is False
    
    def test_validate_operator(self):
        """Test operator validation."""
        schema = BRAINDataFields()
        assert schema.validate_operator("ts_rank") is True
        assert schema.validate_operator("rank") is True
        assert schema.validate_operator("group_neutralize") is True
        assert schema.validate_operator("fake_op") is False
    
    def test_validate_expression_basic(self):
        """Test basic expression validation."""
        schema = BRAINDataFields()
        
        # Valid expression
        valid, msg = schema.validate_expression_basic("rank(close)")
        assert valid is True
        
        # Empty expression
        valid, msg = schema.validate_expression_basic("")
        assert valid is False
        
        # Unmatched parentheses
        valid, msg = schema.validate_expression_basic("rank(close")
        assert valid is False
        
        # Dangerous pattern
        valid, msg = schema.validate_expression_basic("rank(close); DROP TABLE")
        assert valid is False
    
    def test_suggest_fields(self):
        """Test field autocomplete."""
        schema = BRAINDataFields()
        suggestions = schema.suggest_fields("ret")
        assert len(suggestions) > 0
        assert any("ret" in s for s in suggestions)
    
    def test_export_schema(self):
        """Test schema export."""
        schema = BRAINDataFields()
        exported = schema.export_schema()
        
        assert "fields" in exported
        assert "operators" in exported
        assert "total_fields" in exported
        assert "total_operators" in exported
        assert len(exported["fields"]) > 0


class TestGlobalDataFields:
    """Test global data fields instance."""
    
    def test_get_data_fields(self):
        """Test getting global instance."""
        schema = get_data_fields()
        assert schema is not None
        assert isinstance(schema, BRAINDataFields)


# Integration test (requires BRAIN credentials)
@pytest.mark.skip(reason="Requires BRAIN credentials in .env")
class TestBRAINSession:
    """Integration tests for BRAIN API session (requires live BRAIN account)."""
    
    @pytest.fixture
    def brain_session(self):
        """Create BRAIN session from env vars."""
        import os
        from dotenv import load_dotenv
        load_dotenv()
        
        email = os.getenv("BRAIN_EMAIL")
        password = os.getenv("BRAIN_PASSWORD")
        
        if not email or not password:
            pytest.skip("BRAIN_EMAIL or BRAIN_PASSWORD not set")
        
        session = BRAINSession(email, password)
        yield session
        session.close()
    
    def test_authenticate(self, brain_session):
        """Test BRAIN authentication."""
        assert brain_session.is_authenticated is True
    
    def test_get_data_fields(self, brain_session):
        """Test fetching data fields from BRAIN."""
        fields = brain_session.get_data_fields()
        assert fields is not None
        assert len(fields) > 1000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
