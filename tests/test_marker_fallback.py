"""Test for Marker parser fallback mechanism."""

from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("marker")  # Skip tests if marker is not installed

from pdfkb.parsers.parser_marker import MarkerPDFParser  # noqa: E402


@pytest.mark.unit
class TestMarkerFallback:
    """Test Marker parser fallback for table recognition errors."""

    @pytest.mark.asyncio
    async def test_marker_fallback_on_tensorlist_error(self, tmp_path):
        """Test that Marker parser falls back when TensorList error occurs."""
        # Create a dummy PDF
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4\ntest content")

        # Create mock rendered output
        mock_rendered = MagicMock()
        mock_rendered.metadata = MagicMock()
        mock_rendered.metadata.page_stats = [1, 2, 3]  # 3 pages

        # Mock the marker imports and components - they're imported inside the parse method
        with (
            patch("marker.config.parser.ConfigParser") as MockConfigParser,
            patch("marker.converters.pdf.PdfConverter") as MockPdfConverter,
            patch("marker.models.create_model_dict") as mock_create_models,
            patch("marker.output.text_from_rendered") as mock_text_from_rendered,
        ):

            # Setup mocks
            mock_create_models.return_value = {}
            mock_text_from_rendered.return_value = ("# Test Content\n\nExtracted text", {}, [])

            # Create a converter that fails first, then succeeds
            mock_converter_instance = MagicMock()

            # Track calls to see if fallback was used
            call_count = [0]
            configs_used = []

            def converter_side_effect(pdf_path):
                call_count[0] += 1
                if call_count[0] == 1:
                    # First call - simulate TensorList error
                    raise RuntimeError("stack expects a non-empty TensorList")
                else:
                    # Second call (fallback) - succeed
                    return mock_rendered

            mock_converter_instance.side_effect = converter_side_effect

            # Track ConfigParser instantiations to verify disable_table_rec
            def config_parser_init(config):
                configs_used.append(config.copy())
                mock_parser = MagicMock()
                mock_parser.generate_config_dict.return_value = {}
                mock_parser.get_processors.return_value = []
                mock_parser.get_renderer.return_value = None
                mock_parser.get_llm_service.return_value = None
                return mock_parser

            MockConfigParser.side_effect = config_parser_init
            MockPdfConverter.return_value = mock_converter_instance

            # Create parser and test
            parser = MarkerPDFParser()
            result = await parser.parse(pdf_file)

            # Verify fallback was triggered
            assert call_count[0] == 2, "Converter should be called twice (original + fallback)"
            assert len(configs_used) == 2, "ConfigParser should be instantiated twice"

            # Verify first config doesn't disable table recognition
            assert "disable_table_rec" not in configs_used[0] or not configs_used[0].get("disable_table_rec")

            # Verify second config (fallback) disables table recognition
            assert configs_used[1].get("disable_table_rec") is True

            # Verify result is correct
            assert result.markdown_content == "# Test Content\n\nExtracted text"
            assert result.metadata["page_count"] == 3

    @pytest.mark.asyncio
    async def test_marker_no_fallback_on_other_errors(self, tmp_path):
        """Test that Marker parser doesn't fallback for non-TensorList errors."""
        # Create a dummy PDF
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4\ntest content")

        # Mock the marker imports - they're imported inside the parse method
        with (
            patch("marker.config.parser.ConfigParser"),
            patch("marker.converters.pdf.PdfConverter") as MockPdfConverter,
            patch("marker.models.create_model_dict") as mock_create_models,
        ):

            mock_create_models.return_value = {}

            # Create a converter that fails with a different error
            mock_converter_instance = MagicMock()
            mock_converter_instance.side_effect = RuntimeError("Some other error")
            MockPdfConverter.return_value = mock_converter_instance

            # Create parser and test
            parser = MarkerPDFParser()

            with pytest.raises(RuntimeError) as excinfo:
                await parser.parse(pdf_file)

            # Verify the error is propagated, not caught
            assert "Failed to parse PDF with Marker" in str(excinfo.value)

            # Verify converter was only called once (no retry)
            assert mock_converter_instance.call_count == 1

    @pytest.mark.asyncio
    async def test_marker_successful_without_fallback(self, tmp_path):
        """Test that Marker parser works normally when no TensorList error occurs."""
        # Create a dummy PDF
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4\ntest content")

        # Create mock rendered output
        mock_rendered = MagicMock()
        mock_rendered.metadata = MagicMock()
        mock_rendered.metadata.page_stats = [1, 2]  # 2 pages

        # Mock the marker imports - they're imported inside the parse method
        with (
            patch("marker.config.parser.ConfigParser") as MockConfigParser,
            patch("marker.converters.pdf.PdfConverter") as MockPdfConverter,
            patch("marker.models.create_model_dict") as mock_create_models,
            patch("marker.output.text_from_rendered") as mock_text_from_rendered,
        ):

            mock_create_models.return_value = {}
            mock_text_from_rendered.return_value = ("# Normal Content\n\nNo errors here", {}, [])

            # Track ConfigParser calls
            configs_used = []

            def config_parser_init(config):
                configs_used.append(config.copy())
                mock_parser = MagicMock()
                mock_parser.generate_config_dict.return_value = {}
                mock_parser.get_processors.return_value = []
                mock_parser.get_renderer.return_value = None
                mock_parser.get_llm_service.return_value = None
                return mock_parser

            MockConfigParser.side_effect = config_parser_init

            # Create a converter that succeeds immediately
            mock_converter_instance = MagicMock()
            mock_converter_instance.return_value = mock_rendered
            MockPdfConverter.return_value = mock_converter_instance

            # Create parser and test
            parser = MarkerPDFParser()
            result = await parser.parse(pdf_file)

            # Verify no fallback was triggered
            assert mock_converter_instance.call_count == 1, "Converter should only be called once"
            assert len(configs_used) == 1, "ConfigParser should only be instantiated once"

            # Verify table recognition was not disabled
            assert "disable_table_rec" not in configs_used[0] or not configs_used[0].get("disable_table_rec")

            # Verify result
            assert result.markdown_content == "# Normal Content\n\nNo errors here"
            assert result.metadata["page_count"] == 2
