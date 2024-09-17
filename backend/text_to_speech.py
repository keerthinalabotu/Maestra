from gtts import gTTS
import os
import re

def extract_response(initial_prompt):
    print("running")
    # pattern = r'<response_to_student_final>\s*(.*?)\s*</response_to_student_final>'
    start = "The response to student"
    start_tag = "<response_to_student_final>"
    end_tag = "</response_to_student_final>"

    start_num = initial_prompt.find(start)
    if start_num == -1:
        print("Start tag not found in the initial prompt")
        return None
    
    start_num += len(start)

    print(initial_prompt[start_num:])
    
    tag_start = initial_prompt.find(start_tag, start_num)
    if tag_start == -1:
        print("Start tag not found after the start phrase")
        return None
    
    # Find the end tag after the start tag
    tag_end = initial_prompt.find(end_tag, tag_start + len(start_tag))
    if tag_end == -1:
        print("End tag not found after the start tag")
        return None
    
    # Extract the content between the tags
    content_start = tag_start + len(start_tag)
    extracted_content = initial_prompt[content_start:tag_end].strip()
    
    print(f"Extracted content: {extracted_content}")
    return extracted_content
    
    # first_start = initial_prompt.find(start_tag)
    # if first_start == -1:
    #     print("Start tag not found in the initial prompt")
    #     return None
    
    # # Find the second occurrence
    # second_start = initial_prompt.find(start_tag, first_start + len(start_tag))
    # if second_start == -1:
    #     print("Second start tag not found in the initial prompt")
    #     return None
    
    # # Find the end tag after the second start tag
    # end_index = initial_prompt.find(end_tag, second_start)
    # if end_index == -1:
    #     print("End tag not found after second start tag")
    #     return None
    
    # # Extract the content between the second pair of tags
    # start_index = second_start + len(start_tag)
    # extracted_content = initial_prompt[start_index:end_index].strip()
    
    # print(f"Extracted content: {extracted_content}")
    # return extracted_content

    # start_index = initial_prompt.find(start_tag)
    # end_index = initial_prompt.find(end_tag)
    
    # if start_index != -1 and end_index != -1:
    #     # Add the length of the start tag to get the beginning of the content
    #     start_index += len(start_tag)
    #     return initial_prompt[start_index:end_index].strip()
    # else:
    #     print("Tags not found in the initial prompt")
    #     return None
    
    # print(pattern)
    # print("pattern fine")
    # match = re.search(pattern, initial_prompt, re.DOTALL)
    # print(match)
    # print("search fine")
    # if match:
    #     print("match fine")
    #     return match.group(1).strip()
    # return None

def text_to_speech(text, lang='en', output_dir='static/audio/'):
    tts = gTTS(text=text, lang=lang)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    filename = f"speech_{hash(text)}.mp3"
    filepath = os.path.join(output_dir, filename)
    
    tts.save(filepath)
    return filepath 