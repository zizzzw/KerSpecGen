import os
import re
import json
import time
import argparse
from http import HTTPStatus
from dashscope import Generation
import dashscope

def replace_alternative_patterns(content):
    """
    Replace alternative section patterns with a standardized format.
    e.g., section "name" -> section <open>name<close>
    """
    content = re.sub(r'(?<!\w)section\s*"([^\"]*?)"', r'section <open>\1<close>', content, flags=re.DOTALL)
    content = re.sub(r'(?<!\w)subsection\s*"([^\"]*?)"', r'subsection <open>\1<close>', content, flags=re.DOTALL)
    content = re.sub(r'(?<!\w)subsubsection\s*"([^\"]*?)"', r'subsubsection <open>\1<close>', content, flags=re.DOTALL)
    return content

def parse_thy_file_by_sections(file_path):
    """
    Parses a .thy file and splits its content by sections, subsections, and subsubsections.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    content = replace_alternative_patterns(content)
    
    section_pattern = re.compile(r'(?<!\w)section\s*\\?<open>(.*?)\\?<close>', re.DOTALL)
    subsection_pattern = re.compile(r'(?<!\w)subsection\s*\\?<open>(.*?)\\?<close>', re.DOTALL)
    subsubsection_pattern = re.compile(r'(?<!\w)subsubsection\s*\\?<open>(.*?)\\?<close>', re.DOTALL)

    results = []
    current_section = ""
    current_subsection = ""
    current_subsubsection = ""

    sections = section_pattern.split(content)
    for i in range(0, len(sections), 2):
        if i > 0:
            match = section_pattern.search(content[len(''.join(sections[:i])):])
            if match:
                current_section = match.group(1).strip()

        subsections = subsection_pattern.split(sections[i])
        for j in range(0, len(subsections), 2):
            if j > 0:
                match = subsection_pattern.search(sections[i][len(''.join(subsections[:j])):])
                if match:
                    current_subsection = match.group(1).strip()

            subsubsections = subsubsection_pattern.split(subsections[j])
            for k in range(0, len(subsubsections), 2):
                if k > 0:
                    match = subsubsection_pattern.search(subsections[j][len(''.join(subsubsections[:k])):])
                    if match:
                        current_subsubsection = match.group(1).strip()

                code = subsubsections[k].strip()
                if code:
                    results.append({
                        "title": file_path,
                        "section": current_section,
                        "subsection": current_subsection,
                        "subsubsection": current_subsubsection,
                        "code": '\n' + code
                    })

    return results

def save_to_jsonl(data, output_file):
    """
    Saves a list of dictionaries to a JSONL file.
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in data:
            json.dump(entry, f)
            f.write('\n')

def run_split_by_section(directory, output_file):
    """
    Processes all .thy files in a directory, splits them by sections,
    and saves the result to a JSONL file.
    """
    print(f"Processing .thy files in directory: {directory}")
    all_results = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.thy'):
                file_path = os.path.join(root, file)
                all_results.extend(parse_thy_file_by_sections(file_path))
    
    save_to_jsonl(all_results, output_file)
    print(f"Saved section-split data to {output_file}")


def run_split_by_comment(input_file, output_file):
    """
    Processes a JSONL file, splitting code blocks by comments.
    """
    print(f"Splitting code by comments from {input_file}")
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        chapter = ""
        lines = infile.readlines()
        for i in range(len(lines)):
            data = json.loads(lines[i])
            pre_title = json.loads(lines[i-1])['title'] if i > 0 else ""
            next_title = json.loads(lines[i+1])['title'] if i < len(lines) - 1 else ""

            if 'subsubsection' in data:
                del data['subsubsection']
            
            if 'code' in data:
                # Comments in (*...*) format often contain irrelevant info like Copyright, so remove them.
                data['code'] = re.sub(r'\(\*(.*?)(Copyright|FIXME|TODO)(.*?)\*\)', '', data['code'], flags=re.DOTALL)
                
                # Regex to extract comments and split the code field
                pattern = r'(?:\n\s*)(\(\*.*?\*\)|text\s*\\<open>.*?\\<close>|<comment>\s*\\<open>.*?\\<close>)'
                
                parts = re.split(pattern, data['code'].strip(), flags=re.DOTALL)                
                
                for j, part in enumerate(parts):
                    part = part.strip()
                    if j % 2 == 0: # This is a code block
                        code_part = part
                        if not code_part:
                            continue

                        comment_part = parts[j-1].strip() if j > 0 else ""
                        comment_text = ""
                        if comment_part:
                            comment_match = re.search(r'\(\*(.*?)\*\)|text\s*\\<open>(.*?)\\<close>|<comment>\s*\\<open>(.*?)\\<close>', comment_part, flags=re.DOTALL)
                            if comment_match:
                                comment_text = comment_match.group(1) or comment_match.group(2) or comment_match.group(3)

                        if code_part.startswith('theory') and code_part.endswith('begin'):
                            continue
                        
                        # Extract chapter
                        match = re.search(r'chapter(?:\s*"|\s*\\<open>)(.*?)(?:\s*"|\s*\\<close>)', code_part)
                        cur_chapter = match.group(1) if match else ""
                        if cur_chapter:
                            chapter = cur_chapter

                            # Special case: if the file has only one code block, keep it.
                            if len(parts) == 1 and data['title'] != next_title:
                                new_data = data.copy()
                                new_data['code'] = code_part
                                new_data['comment'] = comment_text
                                new_data['chapter'] = chapter
                                assert new_data['code'] != ""
                                outfile.write(json.dumps(new_data, ensure_ascii=False) + '\n')

                            if comment_text:
                                chapter += f"\n(*{comment_text}*)"
                            continue

                        if data['title'] != pre_title:
                            chapter = ""

                        new_data = data.copy()
                        new_data['code'] = code_part
                        new_data['comment'] = comment_text
                        new_data['chapter'] = chapter
                        assert new_data['code'] != ""
                        outfile.write(json.dumps(new_data, ensure_ascii=False) + '\n')
    print(f"Saved comment-split data to {output_file}")


def run_split_by_keyword(input_file, output_file):
    """
    Processes a JSONL file, splitting code blocks by Isabelle keywords.
    This logic was commented out in the original notebook.
    """
    print(f"Splitting code by keywords from {input_file}")
    # This function is based on the commented out section in the notebook.
    # It might be an alternative way of processing.
    keywords = [
        'definition', 'fun', 'primrec', 'termination', 'context', 'setup', 
        'locale', 'local_setup', 'local', 'record', 'axiomatization', 
        'type_synonym', 'declare', 'unbundle', 'crunch_ignore', 'crunch', 
        'requalify_facts', 'global_interpretation', 'inductive_set', 
        'crunches', 'lemmas', 'lemma', 'method', 'inductive', 
        'match_abbreviation', 'abbreviation'
    ]
    split_pattern = r'(?<=\n)(' + '|'.join(keywords) + r')(?=\s)'

    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            data = json.loads(line)
            if 'code' not in data:
                continue

            # This part of logic is a direct translation from the commented-out notebook cell.
            # Its interaction with the previous splitting step (by comment) should be considered.
            # For simplicity, we assume this is an alternative to `run_split_by_comment`.
            code_part = data['code']
            comment = data.get('comment', '') # Assuming comment is already extracted

            sub_parts = re.split(split_pattern, code_part)
            sub_parts = [p.strip() for p in sub_parts if p.strip()]

            if not sub_parts:
                continue

            # This logic for recombining is complex and directly from the notebook.
            if sub_parts and sub_parts[0] in keywords:
                for i in range(0, len(sub_parts), 2):
                    new_data = data.copy()
                    new_data['code'] = sub_parts[i] + ' ' + (sub_parts[i+1] if i+1 < len(sub_parts) else '')
                    new_data['comment'] = comment if i == 0 else ""
                    if new_data['code']:
                        outfile.write(json.dumps(new_data, ensure_ascii=False) + '\n')
            else:
                # This seems to handle code that does not start with a keyword.
                new_data = data.copy()
                new_data['code'] = sub_parts[0]
                new_data['comment'] = comment
                if new_data['code']:
                    outfile.write(json.dumps(new_data, ensure_ascii=False) + '\n')

                for i in range(1, len(sub_parts), 2):
                    new_data = data.copy()
                    new_data['code'] = sub_parts[i] + ' ' + (sub_parts[i+1] if i+1 < len(sub_parts) else '')
                    new_data['comment'] = "" # Comment only for the first part
                    if new_data['code']:
                        outfile.write(json.dumps(new_data, ensure_ascii=False) + '\n')

    print(f"Saved keyword-split data to {output_file}")


def run_filter_records_with_comments(input_file, output_file):
    """
    Filters a JSONL file to keep only records with non-empty comments.
    """
    print(f"Filtering records with non-empty comments from {input_file}")
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            try:
                data = json.loads(line)
                if 'comment' in data and data['comment'].strip() != "":
                    outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
                continue
    print(f"Saved filtered records to {output_file}")


def generate_property_for_chunk(line, api_key, model):
    """
    Calls an LLM to generate a property for a chunk of code.
    """
    dashscope.api_key = api_key
    data = json.loads(line)
    
    # Unpacking data for prompt
    title = data.get('title', '')
    chapter = data.get('chapter', '')
    section = data.get('section', '')
    subsection = data.get('subsection', '')
    comment = data.get('comment', '')
    code = data.get('code', '')

    user_input = """In operating system formalization, a specification is a detailed, formal description of the system's behavior, requirements, and constraints in code, serving as a blueprint for implementation and verification. Property should be abstract and concise in 1-3 sentence, you can learn from the comment if it is provided. 
Your task is to summarize specification property from code. You should answer question with one main property and optional subproperties(if the code is simple and one main property is enough to summarize, subproperties are unnecessary), without any extra explanation or statement like "Here are the prorperties:". Please note that do not use the parts or fuction name in code to explain, just extract properties like that they are defined before writing spec code.
Here are examples:

Example1:
section: Thread Message Formats
comment: TCB capabilities confer authority to perform seven actions. A thread can request to yield its timeslice to another, to suspend or resume another, to reconfigure another thread, or to copy register sets into, out of or between other threads.
code:
fun
  invoke_tcb :: "tcb_invocation ⇒ (data list,'z::state_ext) p_monad"
where
  "invoke_tcb (Suspend thread) = liftE (do suspend thread; return [] od)"
| "invoke_tcb (Resume thread) = liftE (do restart thread; return [] od)"
... (rest of the code from notebook) ...
Answer 1:
TCB Invocation Actions: Allows threads to perform actions such as suspending and resuming other threads, controlling various thread settings (fault endpoints, priorities, capability roots, IPC buffers), copying registers between threads, reading and writing thread registers, controlling notifications (binding and unbinding), and setting the TLS base. These actions ensure that threads can manage their state and capabilities effectively, maintaining the integrity and performance of the kernel.
Domain Management: Set the scheduling domain for a thread. It ensures that the thread is correctly dequeued, its domain is updated, and it is re-enqueued if runnable. If the thread being updated is the current thread, a reschedule is required. This ensures that threads are correctly managed within their scheduling domains, optimizing the kernel's scheduling and execution efficiency.

Example 2:
section: IPC Capability Transfers
comment: In addition to the data payload a message may also contain capabilities. When a thread requests additional capabilities be transferred the identities of those capabilities are retreived from the thread's IPC buffer.
code:
definition
  buffer_cptr_index :: nat
where
 "buffer_cptr_index ≡ (msg_max_length + 2)"
... (rest of the code from notebook) ...
Answer 2:
Get Extra Capability Pointers: Retrieve the additional capability pointers from the thread's IPC buffer. Load the word offsets corresponding to the extra capabilities and convert them to capability pointers.

Now, summarize property based on the following information. Don't use the information from the given example to summarize property.
"""
    if chapter:
        user_input += f"title:\n{chapter}\n\n"
    if section:
        user_input += f"section:\n{section}\n\n"
    if subsection:
        user_input += f"subsection:\n{subsection}\n\n"
    if comment:
        user_input += f"comment:\n{comment}\n\n"
    user_input += f"code:\n{code}"

    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant in operating system property generation.'},
        {'role': 'user', 'content': user_input}
    ]

    response = Generation.call(model=model, messages=messages, result_format='message')

    if response.status_code == HTTPStatus.OK:
        assistant_message = response.output.choices[0]['message']
        return {
            'spec': code,
            'property': assistant_message['content'],
            'title': title,
            'chapter': chapter,
            'section': section,
            'comment': comment
        }
    else:
        print(f"Request id: {response.request_id}, Status code: {response.status_code}, error code: {response.code}, error message: {response.message}")
        return None

def run_property_generation(input_file, output_file, model, api_keys, start_index=0, sleep_time=3):
    """
    Generates properties for code chunks using an LLM.
    """
    if not api_keys:
        print("API keys are required for property generation.")
        return

    print(f"Generating properties using model {model} from {input_file}")
    with open(input_file, 'r', encoding='utf-8') as file:
        json_lines = file.readlines()

    with open(output_file, 'a', encoding='utf-8') as outfile:
        i = 0
        for line in json_lines[start_index:]:
            key = api_keys[i % len(api_keys)]
            result = generate_property_for_chunk(line, key, model)
            if result:
                json.dump(result, outfile, ensure_ascii=False)
                outfile.write('\n')
                i += 1
                time.sleep(sleep_time)
            else:
                print("Stopping due to API error.")
                break
    print(f"Finished generating properties. Output at {output_file}")


def clean_generated_property(data, api_key):
    """
    Cleans up generated property text by removing introductory phrases.
    """
    dashscope.api_key = api_key
    prompt_text = f'''You are a data cleaning assistant. Your task is to remove introductory phrases like "Property:" from the text, retaining only the actual property descriptions.
Text 1: Property: Define general-purpose registers within the machine architecture.
Answer 1: Define general-purpose registers within the machine architecture.
Text 2: Property 1: Updates access rights of architecture capabilities while validating virtual memory rights.\\nProperty 2: Defines architecture-specific object types including various page sizes and table structures.\\nProperty 3: Represents interrupt request (IRQ) states for the x64 architecture with details for MSI and IOAPIC interrupts.
Answer 2: Updates access rights of architecture capabilities while validating virtual memory rights.Defines architecture-specific object types including various page sizes and table structures.Represents interrupt request (IRQ) states for the x64 architecture with details for MSI and IOAPIC interrupts.
Text to clean: {data['property']}
Please provide only the cleaned properties, without any additional content, introductions, or conclusions.
    '''
    response = Generation.call(model='qwen2-1.5b-instruct',
                            prompt=prompt_text,
                            result_format='message')
    if response.status_code == HTTPStatus.OK:
        assistant_message = response.output.choices[0]['message']
        data['property'] = assistant_message['content']
        return data
    else:
        print(f"Request id: {response.request_id}, Status code: {response.status_code}, error code: {response.code}, error message: {response.message}")
        return None

def run_property_cleaning(input_file, output_file, api_keys, start_index=0, sleep_time=0.5):
    """
    Cleans the 'property' field in a JSONL file.
    """
    if not api_keys:
        print("API keys are required for property cleaning.")
        return

    print(f"Cleaning properties from {input_file}")
    with open(input_file, 'r', encoding='utf-8') as file:
        json_lines = file.readlines()

    with open(output_file, 'a', encoding='utf-8') as outfile:
        count = 0
        for line in json_lines[start_index:]:
            api_key = api_keys[count % len(api_keys)]
            data = json.loads(line)
            cleaned_data = clean_generated_property(data, api_key)
            if cleaned_data:
                json.dump(cleaned_data, outfile, ensure_ascii=False)
                outfile.write('\n')
                count += 1
                time.sleep(sleep_time)
    print(f"Finished cleaning properties. Output at {output_file}")

def run_remove_duplicate_comments(input_file, output_file):
    """
    Removes consecutive duplicate comments from a JSONL file.
    """
    print(f"Removing duplicate consecutive comments from {input_file}")
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        previous_comment = None
        for line in infile:
            data = json.loads(line)
            if 'comment' in data:
                current_comment = data['comment']
                if current_comment == previous_comment:
                    data['comment'] = ''
                previous_comment = current_comment
            outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
    print(f"Saved cleaned data to {output_file}")


def filter_comment_by_relevance(data, api_key):
    """
    Filters comments based on their relevance to the property.
    """
    dashscope.api_key = api_key
    prompt_text = f'''You are an operating system assistant. Please determine if the 'comment' contains most of the information present in the 'property'. Answer 'yes' if it is relevant, and 'no' otherwise.
Text 1:
Property: Establishes the foundation for defining access rights within the system.
Comment: Definition of access rights.
Answer 1: yes

Text 2:
Property: Derive capabilities into a copyable form, handling various types with special cases, while ensuring no child capabilities exist for untyped caps, and returning null for non-copyable types.
Comment: Derive a cap into a form in which it can be copied. For internal reasons not all capability types can be copied at all times and not all capability types can be copied unchanged.
Answer 2: yes

Text 3:
Property: Define a constant identifier 'id0' with the value 0.
Comment: Proof to be done.
Answer 3: no

Text 4:
Property: Determine if a capability slot is the direct parent of another in the capability derivation tree.
Comment: to be used for abstraction unifying kernel objects other than TCB and CNode
Answer 4: no

Content to evaluate:
Property: {data['property']}
Comment: {data['comment']}

Please answer only 'yes' or 'no', without any other content, introductions, or conclusions.
    '''
    response = Generation.call(model='qwen-turbo',
                            prompt=prompt_text,
                            result_format='message')
    if response.status_code == HTTPStatus.OK:
        assistant_message = response.output.choices[0]['message']
        if assistant_message['content'].strip().lower() == 'no':
            data['comment'] = ''
        return data
    else:
        print(f"Request id: {response.request_id}, Status code: {response.status_code}, error code: {response.code}, error message: {response.message}")
        return None

def run_relevance_filtering(input_file, output_file, api_keys, sleep_time=0.5):
    """
    Filters comments in a JSONL file based on relevance to properties.
    """
    if not api_keys:
        print("API keys are required for relevance filtering.")
        return
        
    print(f"Filtering comments by relevance from {input_file}")
    with open(input_file, 'r', encoding='utf-8') as file:
        json_lines = file.readlines()
    
    with open(output_file, 'a', encoding='utf-8') as outfile:
        count = 0
        for line in json_lines:
            data = json.loads(line)
            if data.get('comment', '').strip():
                api_key = api_keys[count % len(api_keys)]
                result_data = filter_comment_by_relevance(data, api_key)
                if result_data:
                    json.dump(result_data, outfile, ensure_ascii=False)
                    outfile.write('\n')
                count += 1
                time.sleep(sleep_time)
            else:
                json.dump(data, outfile, ensure_ascii=False)
                outfile.write('\n')
    print(f"Finished relevance filtering. Output at {output_file}")


def main():
    parser = argparse.ArgumentParser(description="A pipeline for extracting properties from Isabelle/HOL specifications.")
    parser.add_argument('steps', nargs='+', help="Which steps to run. Available: split_section, split_comment, split_keyword, filter_comment, gen_prop, clean_prop, dedup_comment, filter_relevance")
    
    # File paths
    parser.add_argument('--spec_dir', default='./spec', help="Directory with .thy files.")
    parser.add_argument('--split_section_out', default='split_by_section.jsonl', help="Output for split_section step.")
    parser.add_argument('--split_comment_out', default='split_by_comment.jsonl', help="Output for split_comment step.")
    parser.add_argument('--split_keyword_out', default='split_by_keyword.jsonl', help="Output for split_keyword step.")
    parser.add_argument('--filter_comment_out', default='record_with_comment.jsonl', help="Output for filter_comment step.")
    parser.add_argument('--gen_prop_out', default='output.jsonl', help="Output for gen_prop step.")
    parser.add_argument('--clean_prop_out', default='cleaned_output.jsonl', help="Output for clean_prop step.")
    parser.add_argument('--dedup_comment_out', default='cleaned_output2.jsonl', help="Output for dedup_comment step.")
    parser.add_argument('--filter_relevance_out', default='cleaned_output3.jsonl', help="Output for filter_relevance step.")
    
    # API and Model settings
    parser.add_argument('--api_keys', nargs='*', default=[], help="API keys for LLM calls.")
    parser.add_argument('--gen_model', default='llama3.1-405b-instruct', help="Model for property generation.")
    
    # Other settings
    parser.add_argument('--start_index', type=int, default=0, help="Line number to start processing from for certain steps.")

    args = parser.parse_args()

    # Define file dependencies
    step_inputs = {
        'split_comment': args.split_section_out,
        'split_keyword': args.split_section_out,
        'filter_comment': args.split_keyword_out, # As in notebook
        'gen_prop': args.split_keyword_out, # As in notebook
        'clean_prop': args.gen_prop_out,
        'dedup_comment': args.clean_prop_out,
        'filter_relevance': args.dedup_comment_out
    }

    for step in args.steps:
        if step == 'split_section':
            run_split_by_section(args.spec_dir, args.split_section_out)
        elif step == 'split_comment':
            run_split_by_comment(step_inputs[step], args.split_comment_out)
        elif step == 'split_keyword':
            run_split_by_keyword(step_inputs[step], args.split_keyword_out)
        elif step == 'filter_comment':
            run_filter_records_with_comments(step_inputs[step], args.filter_comment_out)
        elif step == 'gen_prop':
            run_property_generation(step_inputs[step], args.gen_prop_out, args.gen_model, args.api_keys, args.start_index)
        elif step == 'clean_prop':
            run_property_cleaning(step_inputs[step], args.clean_prop_out, args.api_keys, args.start_index)
        elif step == 'dedup_comment':
            run_remove_duplicate_comments(step_inputs[step], args.dedup_comment_out)
        elif step == 'filter_relevance':
            run_relevance_filtering(step_inputs[step], args.filter_relevance_out, args.api_keys)
        else:
            print(f"Unknown step: {step}")

if __name__ == '__main__':
    main()
