import os

def get_buffer(force_iTerm=False):
    # Check if running in iTerm2
    term_program = os.getenv("TERM_PROGRAM", "")

    current_file_dir = os.path.dirname(os.path.realpath(__file__))
    
    if term_program == "iTerm.app" or force_iTerm:
        # Proceed with iTerm2 specific logic
        # You can run the AppleScript to fetch the buffer
        os.system(f'osascript {current_file_dir}/get_iterm_buffer.applescript')

        home_dir = os.getenv("HOME")
        with open(f'{home_dir}/iterm_buffer.txt', 'r') as file:
            buffer = file.read()
            return buffer
    else:
        print("Not running in iTerm2")
    
    return None