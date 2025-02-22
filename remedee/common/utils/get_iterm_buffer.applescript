-- AppleScript to fetch iTerm2 scrollback buffer
tell application "iTerm2"
    tell current session of current tab of current window
        set scrollback_content to get contents
    end tell
end tell

-- Define the output file path
set output_file to (POSIX path of (path to home folder)) & "iterm_buffer.txt"

-- Write the scrollback content to the file
set file_handle to open for access POSIX file output_file with write permission
write scrollback_content to file_handle
close access file_handle
