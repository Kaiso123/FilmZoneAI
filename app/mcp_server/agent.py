import asyncio
import os
from typing import Optional, List, Dict
from contextlib import AsyncExitStack

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client  # For HTTP transport
from google.generativeai import GenerativeModel, configure
from dotenv import load_dotenv

load_dotenv()  

class MCPAgent:
    def __init__(self, server_url: str = "http://127.0.0.1:4231/mcp"):
        self.server_url = server_url
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        # Configure Gemini API
        configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = GenerativeModel("models/gemini-2.0-flash")  
        self.available_tools: List[Dict] = []

    async def connect_to_server(self):
        """Connect to the MCP server via HTTP"""
        http_transport = await self.exit_stack.enter_async_context(streamablehttp_client(self.server_url))
        read, write, get_session_id_callback = http_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(read, write))

        await self.session.initialize()

        # List available tools and store their metadata
        response = await self.session.list_tools()
        self.available_tools = [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema if hasattr(tool, 'inputSchema') else {}
            }
            for tool in response.tools
        ]
        print("\nConnected to server with tools:", [tool["name"] for tool in self.available_tools])

    async def process_query(self, query: str) -> str:
        """Process user query: Use Gemini to select tool and generate inputs, then call the tool"""
        if not self.session:
            raise ValueError("Session not initialized. Call connect_to_server first.")

        # Prepare prompt for Agent to decide on tool and inputs
        system_prompt = (
            "You are an AI agent that selects the appropriate tool based on the user query. "
            "Available tools:\n" + "\n".join([f"- {tool['name']}: [{tool['description']},\n {tool['input_schema']}]" for tool in self.available_tools]) + "\n"
            "For the query, output ONLY a JSON object with: "
            "{'tool_name': 'selected_tool', 'inputs': {json_inputs_for_tool}}"
            "If no tool matches, set 'tool_name' to None."
        )
        prompt = f"Query: {query}\nSelect tool and generate inputs."

        # Call model API to get tool selection and inputs
        response = self.model.generate_content(
            [system_prompt, prompt],
            generation_config={"max_output_tokens": 500}
        )
        llm_output = response.text.strip()
        print("\nGemini Output:", llm_output)

        try:
            print(1)
            decision = eval(llm_output)  
            tool_name = decision.get("tool_name")
            print(2)
            inputs = decision.get("inputs", {})
            print(3)
        except Exception as e:
            return f"Error parsing Gemini decision: {str(e)}"

        if not tool_name or tool_name not in [tool["name"] for tool in self.available_tools]:
            return "No suitable tool found for the query."

        # Call the selected tool
        try:
            result = await self.session.call_tool(tool_name, inputs)
            return result.content  
        except Exception as e:
            return f"Error calling tool {tool_name}: {str(e)}"

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()

async def main():
    agent = MCPAgent()
    try:
        await agent.connect_to_server()
        
        # Interactive loop for testing
        print("\nMCP Agent Started! Type 'quit' to exit.")
        while True:
            query = input("\nUser Query: ").strip()
            if query.lower() == 'quit':
                break
            response = await agent.process_query(query)
            print("\nResponse:", response)
    finally:
        await agent.cleanup()

if __name__ == "__main__":
    asyncio.run(main())