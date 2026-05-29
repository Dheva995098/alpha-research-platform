#!/usr/bin/env python
"""
Phase 1 Verification Script
Validates that all core modules load correctly.
"""
import sys
import traceback

def test_imports():
    """Test all Phase 1 module imports."""
    tests = [
        ("Config", lambda: __import__('backend.config', fromlist=['settings'])),
        ("Security", lambda: __import__('backend.security', fromlist=['encrypt_credential', 'decrypt_credential', 'generate_aes_key'])),
        ("BRAIN API", lambda: __import__('backend.core.brain_api', fromlist=['BRAINSession', 'BRAINClient'])),
        ("Data Fields", lambda: __import__('backend.core.data_fields', fromlist=['BRAINDataFields', 'get_data_fields'])),
        ("Models", lambda: __import__('backend.models', fromlist=['Account', 'Simulation', 'Result', 'init_db'])),
        ("FastAPI App", lambda: __import__('backend.main', fromlist=['app'])),
    ]
    
    print("=" * 60)
    print("Phase 1 Verification - Module Imports")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            test_func()
            print(f"[OK] {test_name:20} - OK")
            passed += 1
        except Exception as e:
            print(f"[FAIL] {test_name:20} - FAILED: {e}")
            traceback.print_exc()
            failed += 1
    
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0

def test_data_fields_validation():
    """Test data fields validation."""
    from backend.core.data_fields import BRAINDataFields
    
    print("\nTesting Data Fields Validation...")
    schema = BRAINDataFields()
    
    tests = [
        ("Field validation: 'close'", lambda: schema.validate_field("close"), True),
        ("Field validation: 'fake'", lambda: schema.validate_field("fake"), False),
        ("Operator validation: 'rank'", lambda: schema.validate_operator("rank"), True),
        ("Operator validation: 'fake_op'", lambda: schema.validate_operator("fake_op"), False),
        ("Expression validation: valid", lambda: schema.validate_expression_basic("rank(close)")[0], True),
        ("Expression validation: injection", lambda: schema.validate_expression_basic("rank(close); DROP")[0], False),
    ]
    
    passed = 0
    for test_name, test_func, expected in tests:
        try:
            result = test_func()
            if result == expected:
                print(f"[OK] {test_name:40} - OK")
                passed += 1
            else:
                print(f"[FAIL] {test_name:40} - Expected {expected}, got {result}")
        except Exception as e:
            print(f"[FAIL] {test_name:40} - Exception: {e}")
    
    return passed == len(tests)

def test_security():
    """Test credential encryption."""
    from backend.security import encrypt_credential, decrypt_credential
    
    print("\nTesting Security (AES-256 Encryption)...")
    
    test_password = "my-secret-password-123"
    
    try:
        encrypted = encrypt_credential(test_password)
        print(f"[OK] Encryption successful (length: {len(encrypted)})")
        
        decrypted = decrypt_credential(encrypted)
        print("[OK] Decryption successful")
        
        if decrypted == test_password:
            print("[OK] Round-trip test passed")
            return True
        else:
            print(f"[FAIL] Round-trip test failed: {decrypted} != {test_password}")
            return False
    except Exception as e:
        print(f"[FAIL] Security test failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print()
    
    # Test 1: Imports
    imports_ok = test_imports()
    
    # Test 2: Data Fields
    try:
        fields_ok = test_data_fields_validation()
    except Exception as e:
        print(f"\n[FAIL] Data fields test failed: {e}")
        fields_ok = False
    
    # Test 3: Security
    try:
        security_ok = test_security()
    except Exception as e:
        print(f"\n[FAIL] Security test failed: {e}")
        security_ok = False
    
    # Summary
    print("\n" + "=" * 60)
    print("PHASE 1 VERIFICATION SUMMARY")
    print("=" * 60)
    print(f"Module Imports:       {'PASS' if imports_ok else 'FAIL'}")
    print(f"Data Fields:          {'PASS' if fields_ok else 'FAIL'}")
    print(f"Security:             {'PASS' if security_ok else 'FAIL'}")
    print("=" * 60)
    
    all_ok = imports_ok and fields_ok and security_ok
    if all_ok:
        print("\nPHASE 1 VERIFICATION: ALL TESTS PASSED")
        print("\nNext steps:")
        print("1. Run: python setup.py")
        print("2. Run: python -m uvicorn backend.main:app --reload")
        print("3. Visit: http://localhost:8000/docs")
    else:
        print("\nPHASE 1 VERIFICATION: SOME TESTS FAILED")
        sys.exit(1)
