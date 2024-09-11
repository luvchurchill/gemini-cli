"""
At the command line, only need to run once to install the package via pip:

$ pip install google-generativeai
"""

import argparse
import google.generativeai as genai
import os
import signal
import sys


def signal_handler(sig, frame):
    """Exit gracefully with ctrl-c"""
    print("\n Bye... ")
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# Set up the model
# I just took these from defaults
generation_config = {
    "temperature": 0.9,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 2048,
}

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]
default_system_prompt = "Answers should be concise unless the user asks for a detailed explanation. For any technical questions, assume the user has general knowledge in the area and just wants an answer to the question he asked. Keep answers short and correct."

fastai_prompt = """You are an autoregressive language model that has been fine-tuned with instruction-tuning and RLHF. You carefully provide accurate, factual, thoughtful,nuanced answers, and are brilliant at reasoning. If you think there might not be a correct answer, you say so.

Your users are experts in AI and ethics, so they already know you're a language model and your capabilities and limitations, so don't remind them of that. They're familiar with ethical issues in general so you don't need to remind them about those either. Don't be verbose in your answers, but do provide details and examples where it might help the explanation. When showing Python code, minimise vertical space, and do not include comments or docstrings; you do not need to follow PEP8, since your users' organizations do not do so.

Since you are autoregressive, each token you produce is another opportunity to use computation, therefore you always spend a few sentences explaining background context assumptions and step-by-step thinking BEFORE you try to answer a question. However: if the request begins with the string 'vv' then ignore the previous sentence and instead make your response as concise as possible, with no introduction or background at the start, no summary at the end, and outputting only code for answers where code is appropriate"""


def main():
    parser = argparse.ArgumentParser(description="Use Googles Gemini through the API")
    parser.add_argument(
        "-p", "--prompt", nargs="+", type=str, help="The prompt to send to the model"
    )
    parser.add_argument(
        "-s", "--system-prompt", nargs="+", type=str, help="Add a custom system prompt"
    )
    args = parser.parse_args()

    system_prompt = (
        " ".join(args.system_prompt) if args.system_prompt else default_system_prompt
    )
    # "gemini-1.5-pro-latest"
    # gemini-1.5-pro-exp-0801
    # gemini-1.5-pro
    model = genai.GenerativeModel(
        model_name="gemini-1.5-pro-exp-0827",
        generation_config=generation_config,
        system_instruction=system_prompt,
        safety_settings=safety_settings,
    )
    convo = model.start_chat(history=[])

    if args.prompt:
        convo.send_message(" ".join(args.prompt))
        print(f"\n{convo.last.text}")
    else:
        interactive_loop(convo)


def interactive_loop(convo):
    while True:
        user_input = input("Input: ")
        if user_input.startswith("/"):
            parts = user_input[1:].split()
            command = parts[0]
            arguments = parts[1:]

            if command == "file":
                try:
                    with open(arguments[0], "r") as f:
                        file_content = f.read()
                        convo.send_message(file_content)
                        print(f"\n{convo.last.text}")

                except FileNotFoundError:
                    print("File not found. ")
            elif command == "exit":
                break
        else:
            convo.send_message(user_input)
            print(f"\n{convo.last.text}")


if __name__ == "__main__":
    main()


