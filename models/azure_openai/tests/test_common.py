import os
import unittest
from unittest.mock import MagicMock, patch

import openai

from models.common import _client_cache, _CommonAzureOpenAI
from models.constants import AZURE_OPENAI_API_VERSION


class CommonAzureOpenAITestCase(unittest.TestCase):
    def setUp(self):
        _client_cache.clear()
        self.addCleanup(_client_cache.clear)
        self.enterContext(patch.dict(os.environ))
        os.environ.pop("AZURE_OPENAI_READ_TIMEOUT", None)

    def assert_timeout(self, kwargs, read):
        self.assertEqual(
            kwargs["timeout"].as_dict(),
            {"connect": 5.0, "read": read, "write": 10.0, "pool": 315.0},
        )
        self.assertEqual(kwargs["max_retries"], 1)

    def test_default_timeout_components(self):
        kwargs = _CommonAzureOpenAI._to_credential_kwargs(
            {
                "openai_api_base": "https://example-resource.openai.azure.com/",
                "openai_api_key": "test-key",
            }
        )
        self.assert_timeout(kwargs, 300.0)

    def test_valid_read_timeout_values(self):
        for value, expected in [
            ("600", 600.0),
            ("30", 30.0),
            ("0.125", 0.125),
            ("  450.5 \t", 450.5),
        ]:
            with self.subTest(value=value), patch.dict(
                os.environ, {"AZURE_OPENAI_READ_TIMEOUT": value}
            ):
                kwargs = _CommonAzureOpenAI._to_credential_kwargs(
                    {
                        "openai_api_base": "https://example-resource.openai.azure.com/",
                        "openai_api_key": "test-key",
                    }
                )
                self.assert_timeout(kwargs, expected)

    @patch("models.common.openai.OpenAI")
    @patch("models.common.openai.AzureOpenAI")
    def test_invalid_read_timeout_fails_before_sdk_construction(self, azure_sdk, v1_sdk):
        invalid = [
            "",
            " ",
            "\t\n",
            "not-a-number",
            "1 second",
            "0",
            "-0.0",
            "-1",
            "nan",
            "NaN",
            "inf",
            "-inf",
            "Infinity",
            "1e309",
        ]
        for endpoint in [
            "https://example-resource.openai.azure.com/",
            "https://example-resource.openai.azure.com/openai/v1/",
        ]:
            for auth in ["api_key", "entra_id_service_principal"]:
                for value in invalid:
                    with self.subTest(endpoint=endpoint, auth=auth, value=value), patch.dict(
                        os.environ, {"AZURE_OPENAI_READ_TIMEOUT": value}
                    ):
                        with self.assertRaisesRegex(
                            ValueError, "AZURE_OPENAI_READ_TIMEOUT.*positive finite"
                        ):
                            _CommonAzureOpenAI._create_client(
                                {
                                    "openai_api_base": endpoint,
                                    "openai_api_key": "test-key",
                                    "auth_method": auth,
                                    "azure_client_id": "client",
                                    "azure_tenant_id": "tenant",
                                    "azure_client_secret": "secret",
                                },
                                use_cache=False,
                            )
        azure_sdk.assert_not_called()
        v1_sdk.assert_not_called()

    @patch("azure.identity.get_bearer_token_provider")
    @patch("azure.identity.ClientSecretCredential")
    @patch("models.common.openai.OpenAI")
    @patch("models.common.openai.AzureOpenAI")
    def test_configured_timeout_for_endpoints_and_auth(
        self, azure_sdk, v1_sdk, identity, token_provider
    ):
        for is_v1 in [False, True]:
            for auth in ["api_key", "entra_id_service_principal"]:
                with self.subTest(is_v1=is_v1, auth=auth), patch.dict(
                    os.environ, {"AZURE_OPENAI_READ_TIMEOUT": "600.5"}
                ):
                    azure_sdk.reset_mock()
                    v1_sdk.reset_mock()
                    identity.reset_mock()
                    token_provider.reset_mock()
                    credentials = {
                        "openai_api_base": "https://example-resource.openai.azure.com/"
                        + ("openai/v1/" if is_v1 else ""),
                        "openai_api_key": "test-key",
                        "auth_method": auth,
                        "azure_client_id": "client",
                        "azure_tenant_id": "tenant",
                        "azure_client_secret": "secret",
                    }
                    _CommonAzureOpenAI._create_client(credentials, use_cache=False)
                    sdk, unused = (v1_sdk, azure_sdk) if is_v1 else (azure_sdk, v1_sdk)
                    sdk.assert_called_once()
                    unused.assert_not_called()
                    kwargs = sdk.call_args.kwargs
                    self.assert_timeout(kwargs, 600.5)
                    if is_v1:
                        self.assertEqual(kwargs["base_url"], credentials["openai_api_base"])
                        self.assertNotIn("api_version", kwargs)
                        self.assertNotIn("azure_endpoint", kwargs)
                    else:
                        self.assertEqual(
                            kwargs["azure_endpoint"], credentials["openai_api_base"]
                        )
                        self.assertEqual(kwargs["api_version"], AZURE_OPENAI_API_VERSION)
                        self.assertNotIn("base_url", kwargs)
                    if auth == "api_key":
                        self.assertEqual(kwargs["api_key"], "test-key")
                        identity.assert_not_called()
                        token_provider.assert_not_called()
                    else:
                        identity.assert_called_once_with(
                            tenant_id="tenant", client_id="client", client_secret="secret"
                        )
                        if is_v1:
                            token_provider.assert_called_once_with(
                                identity.return_value,
                                "https://cognitiveservices.azure.com/.default",
                            )
                            self.assertIs(kwargs["api_key"], token_provider.return_value)
                        else:
                            self.assertIs(
                                kwargs["azure_ad_token_provider"](),
                                identity.return_value.get_token.return_value.token,
                            )
                            identity.return_value.get_token.assert_called_with(
                                "https://cognitiveservices.azure.com/.default"
                            )
                            self.assertNotIn("api_key", kwargs)

    @patch("models.common.openai.OpenAI")
    @patch("models.common.openai.AzureOpenAI")
    def test_timeout_preserves_cache_and_validation_bypass(self, azure_sdk, v1_sdk):
        for is_v1 in [False, True]:
            with self.subTest(is_v1=is_v1):
                _client_cache.clear()
                sdk = v1_sdk if is_v1 else azure_sdk
                sdk.reset_mock()
                sdk.side_effect = [MagicMock(), MagicMock()]
                credentials = {
                    "openai_api_base": "https://example-resource.openai.azure.com/"
                    + ("openai/v1/" if is_v1 else ""),
                    "openai_api_key": "test-key",
                }
                key = _CommonAzureOpenAI._credential_cache_key(credentials)
                with patch.dict(os.environ, {"AZURE_OPENAI_READ_TIMEOUT": "600"}):
                    cached = _CommonAzureOpenAI._create_client(credentials)
                self.assert_timeout(sdk.call_args.kwargs, 600.0)
                with patch.dict(os.environ, {"AZURE_OPENAI_READ_TIMEOUT": "invalid"}):
                    self.assertIs(_CommonAzureOpenAI._create_client(credentials), cached)
                    with self.assertRaisesRegex(ValueError, "AZURE_OPENAI_READ_TIMEOUT"):
                        _CommonAzureOpenAI._create_client(credentials, use_cache=False)
                self.assertEqual(sdk.call_count, 1)
                with patch.dict(os.environ, {"AZURE_OPENAI_READ_TIMEOUT": "0.25"}):
                    uncached = _CommonAzureOpenAI._create_client(credentials, use_cache=False)
                self.assertIsNot(uncached, cached)
                self.assertEqual(sdk.call_count, 2)
                self.assert_timeout(sdk.call_args.kwargs, 0.25)
                self.assertEqual(_CommonAzureOpenAI._credential_cache_key(credentials), key)
                self.assertEqual(len(_client_cache), 1)
                self.assertIs(_client_cache[key], cached)

    def test_non_v1_endpoint_keeps_api_version(self):
        credentials = {
            "openai_api_base": "https://example-resource.openai.azure.com/",
            "openai_api_key": "test-key",
        }

        kwargs = _CommonAzureOpenAI._to_credential_kwargs(credentials)

        self.assertEqual(
            kwargs["azure_endpoint"],
            "https://example-resource.openai.azure.com/",
        )
        self.assertEqual(kwargs["api_version"], AZURE_OPENAI_API_VERSION)
        self.assertNotIn("base_url", kwargs)
        self.assertIsInstance(kwargs["timeout"], openai.Timeout)

    def test_v1_endpoint_omits_api_version(self):
        credentials = {
            "openai_api_base": "https://example-resource.openai.azure.com/openai/v1/",
            "openai_api_key": "test-key",
            "openai_api_version": "2025-04-01-preview",
        }

        kwargs = _CommonAzureOpenAI._to_credential_kwargs(credentials)

        self.assertEqual(
            kwargs["base_url"],
            "https://example-resource.openai.azure.com/openai/v1/",
        )
        self.assertNotIn("api_version", kwargs)
        self.assertNotIn("azure_endpoint", kwargs)

    @patch("models.common.openai.AzureOpenAI")
    def test_create_client_reuses_cached_client(self, mock_azure_openai):
        mock_client = MagicMock()
        mock_azure_openai.return_value = mock_client
        credentials = {
            "openai_api_base": "https://example-resource.openai.azure.com/",
            "openai_api_key": "test-key",
        }

        first = _CommonAzureOpenAI._create_client(credentials)
        second = _CommonAzureOpenAI._create_client(credentials)

        self.assertIs(first, second)
        mock_azure_openai.assert_called_once()

    @patch("models.common.openai.AzureOpenAI")
    def test_create_client_different_credentials_create_separate_clients(
        self, mock_azure_openai
    ):
        mock_azure_openai.side_effect = [MagicMock(), MagicMock()]
        base_credentials = {
            "openai_api_base": "https://example-resource.openai.azure.com/",
            "openai_api_key": "test-key",
        }

        first = _CommonAzureOpenAI._create_client(base_credentials)
        second = _CommonAzureOpenAI._create_client(
            {**base_credentials, "openai_api_key": "other-key"}
        )

        self.assertIsNot(first, second)
        self.assertEqual(mock_azure_openai.call_count, 2)

    @patch("models.common.openai.AzureOpenAI")
    def test_create_client_use_cache_false_bypasses_cache(self, mock_azure_openai):
        mock_azure_openai.side_effect = [MagicMock(), MagicMock()]
        credentials = {
            "openai_api_base": "https://example-resource.openai.azure.com/",
            "openai_api_key": "test-key",
        }

        first = _CommonAzureOpenAI._create_client(credentials, use_cache=False)
        second = _CommonAzureOpenAI._create_client(credentials, use_cache=False)

        self.assertIsNot(first, second)
        self.assertEqual(mock_azure_openai.call_count, 2)


if __name__ == "__main__":
    unittest.main()
