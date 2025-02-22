import argparse
import logging
import asyncio
import aioconsole
import os
import signal


from remedee.common.utils.sys_info import gather_system_info

logging.basicConfig(filename='remedee_query.log', level=logging.DEBUG)

from remedee.common.llm.llm import LLM
from remedee.common.utils.terminal import get_buffer

def get_recent_output(buffer):
    lines = buffer.split("\n")
    last_command = lines[-10:]
    return "\n".join(last_command)

class Chat:
    def __init__(self):
        self.llm = LLM()
        self.sys_info = gather_system_info()
        user_name = os.getenv("USER")
        self.system_prompt = f"""You are a zsh shell expert and a great software engineer! I am {user_name} and you help me analyze terminal output
or answer generic questions.

My system Info:
        - Operating System: {self.sys_info['Operating System']} ({self.sys_info['OS Version']})
        - Architecture: {self.sys_info['Architecture']}
        - CPU Cores: {self.sys_info['CPU Cores']}
        - Current Directory: {self.sys_info['Current Directory']}
"""
        self.history = [{"role": "system", "content": self.system_prompt}]

    async def request(self, user_prompt) -> str:
        self.history.append({"role": "user", "content": user_prompt})

        partial_response = ""
        async def stream_callback(response_text, token):
            if token:
                nonlocal partial_response 
                if len(partial_response) == 0:
                    print("")
                print(token, end="")
                partial_response = response_text
            return True
        
        try:
            res, _ = await self.llm.query(self.history, stream_callback=stream_callback)
            self.history.append({"role": "assistant", "content": res})
            return res
        except asyncio.CancelledError:
            if len(partial_response):
                self.history.append({"role": "assistant", "content": partial_response + " (response interrupted)"})
            raise  # Re-raise the exception so it can be caught in main

async def main():
    parser = argparse.ArgumentParser(description="Query LLM.")
    parser.add_argument("-p", "--prompt", type=str, default=None, help="Query prompt. Optional if you want to start interactively.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode.")
    args = parser.parse_args()

    buffer = get_buffer(force_iTerm=args.debug)
    if not buffer:
        print("Failed to get buffer.")
        return
    
    prompt = args.prompt.replace("\\n", "\n").strip() if args.prompt else ""
    if len(prompt) == 0:
        prompt = "Hey!"
    else:
        recent_output = get_recent_output(buffer)
        prompt = f"{recent_output}\n\n{prompt}"

    chat = Chat()
    while prompt not in ["exit", "q"]:
        try:
            if len(prompt) == 0:
                prompt = await aioconsole.ainput("👉 ")  # Use aioconsole for non-blocking input

            if len(prompt) > 0:
                _ = await chat.request(prompt)
                print("\n")
        except EOFError:  # Handle Ctrl-D (EOF) to exit gracefully
            print("\nExiting...")
            break
        except KeyboardInterrupt:  # Handle Ctrl-C to exit gracefully
            print("\nInterrupted (Ctrl-D to exit)...")
        except asyncio.CancelledError:
            print("\nRequest canceled (Ctrl-D to exit)...")

        prompt = ""

# def shutdown(loop):
#     """Gracefully shutdown the event loop."""
#     tasks = [t for t in asyncio.all_tasks() if not t.done()]
#     if tasks:
#         for task in tasks:
#             task.cancel()
#         loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
#     loop.stop()

if __name__ == "__main__":
    loop = asyncio.get_event_loop()

    # Setup signal handlers for multiple Ctrl-C presses
    # for sig in (signal.SIGINT, signal.SIGTERM):
    #     loop.add_signal_handler(sig, lambda: shutdown(loop))

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProgram terminated by user.")

