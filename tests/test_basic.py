import pytest

def test_hello_world():
    """Basic hello world test"""
    assert True

def test_addition():
    """Test basic math"""
    assert 1 + 1 == 2

def test_string():
    """Test string operations"""
    message = "Hello, World!"
    assert "Hello" in message
    assert len(message) == 13

def test_list():
    """Test list operations"""
    my_list = [1, 2, 3]
    assert len(my_list) == 3
    assert 2 in my_list

class TestExample:
    """Group related tests in a class"""
    
    def test_upper(self):
        assert "hello".upper() == "HELLO"
    
    def test_lower(self):
        assert "WORLD".lower() == "world"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])