"""Test script to verify LLM API key connections."""

import asyncio
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from app.core.config import settings
from app.services.llm.service import LLMService, LLMProvider


async def test_connections():
    """Test both OpenAI and Gemini API connections."""
    print("\n" + "=" * 60)
    print("Testing LLM API Connections")
    print("=" * 60 + "\n")

    # Check API keys configuration
    print("1. Checking API Keys Configuration:")
    print(f"   ✓ OPENAI_API_KEY: {'Set' if settings.OPENAI_API_KEY else '⚠ Not Set'}")
    print(f"   ✓ GEMINI_API_KEY: {'Set' if settings.GEMINI_API_KEY else '⚠ Not Set'}")
    print()

    # Initialize service
    try:
        service = LLMService()
        print("2. LLMService initialized successfully ✓\n")
    except Exception as e:
        print(f"2. Failed to initialize LLMService ✗")
        print(f"   Error: {e}\n")
        return

    # Test OpenAI
    if settings.OPENAI_API_KEY:
        print("3. Testing OpenAI Connection:")
        try:
            test_prompt = "Say 'OpenAI connection successful' in one sentence."
            response = await service.generate(
                prompt=test_prompt,
                provider=LLMProvider.OPENAI,
                model=settings.OPENAI_MODEL,
            )
            print(f"   ✓ OpenAI Connected")
            print(f"   Response: {response}\n")
        except Exception as e:
            print(f"   ✗ OpenAI Connection Failed")
            print(f"   Error: {str(e)}\n")
    else:
        print("3. Testing OpenAI Connection:")
        print("   ⚠ Skipped (OPENAI_API_KEY not set)\n")

    # Test Gemini
    if settings.GEMINI_API_KEY:
        print("4. Testing Gemini Connection:")
        try:
            test_prompt = "Say 'Gemini connection successful' in one sentence."
            response = await service.generate(
                prompt=test_prompt,
                provider=LLMProvider.GEMINI,
                model=settings.GEMINI_MODEL,
            )
            print(f"   ✓ Gemini Connected")
            print(f"   Response: {response}\n")
        except Exception as e:
            print(f"   ✗ Gemini Connection Failed")
            print(f"   Error: {str(e)}\n")
    else:
        print("4. Testing Gemini Connection:")
        print("   ⚠ Skipped (GEMINI_API_KEY not set)\n")

    print("=" * 60)
    print("Test Complete")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(test_connections())
