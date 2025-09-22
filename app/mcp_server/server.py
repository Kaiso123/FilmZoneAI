import logging
from mcp.server.fastmcp import FastMCP
from ..models.translateModels import TranslateRequest, TranslateResponse, TranslateSegmentsRequest, TranslateSegmentSrt
from ..services.translation_serivce import translate_text as translate_text_service
from ..services.translation_serivce import translate_segments as translate_segments_service

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

def create_mcp_server():
    mcp = FastMCP(
        name="MCP FilmZoneAi Server",
        host="127.0.0.1",
        port=4231,
    )
    ###### TOOLS ######
    @mcp.tool(
        name="translate_text",
        title="Translate Text",
        description="Translate a list of texts from vi to en or en to vi using a specified model. Remember the prefix before the text in the texts." \
        "Example: {'inputs': ['vi: xin chào', 'en: Hello world']} -> ['en: Hello world', 'vi: Xin chào']",
    )
    def translate_text(inputs: TranslateRequest) -> TranslateResponse:
        outputs = translate_text_service(inputs.texts)
        return TranslateResponse(translations=outputs)

    @mcp.tool(
        name="translate_segments",
        title="Translate Segments",
        description="Translate a list of segments from vi to en or en to vi using a specified model.",
    )
    def translate_segments(inputs: TranslateSegmentsRequest) -> TranslateSegmentSrt:
        srt_str, segments = translate_segments_service(inputs.segments, inputs.language)
        return TranslateSegmentSrt(srt=srt_str, segments=segments)
    
    return mcp

def run_mcp_server():
    mcp = create_mcp_server()
    mcp.run(transport="streamable-http")

if __name__ == "__main__":
    run_mcp_server()