print("start")
import os
from rag_model import Generation_model
from rag_model import PromptGeneration
from dotenv import load_dotenv
from config import TEST_FILE

load_dotenv()

def main():
    print("working")

    # test_file_path = r"C:\Users\intelaipc\Downloads\19072018160748.pdf"
    # print(f"Test file path: {test_file_path}")

    # try:
    #     with open(TEST_FILE, 'r') as file:
    #         content = file.read()
    #         print(f"File content: {content[:100]}...")  # Print first 100 characters
    # except Exception as e:
    #     print(f"Error opening file: {e}")

    # current_dir = os.path.dirname(os.path.abspath(TEST_FILE))
    # test_file_path = os.path.join(current_dir, "19072018160748.pdf")

    rag_model = Generation_model()
    prompt_generator = PromptGeneration(rag_model)

    qa = rag_model.process_document(TEST_FILE)

    if qa is None:
        print("Failed to process document. Exiting.")
        return
    
    print("Document processed successfully")

    print("Generating prompts")

    try:
        prompts = prompt_generator.generate_prompts(qa, num_prompts=3)
        print(f"Number of prompts generated: {len(prompts)}")
    except Exception as e:
        print(f"Error in generate_prompts: {e}")
        import traceback
        traceback.print_exc()
        return

    # prompts = prompt_generator.generate_prompts(qa, num_prompts=3)

    print("generated Prompts:")
    for i, prompt in enumerate(prompts, 1):
        print(f"prompt {i}: {prompt}")


if __name__ == "__main__":
    print("Script started")
    main()
    print("Script ended")
