from cat.mad_hatter.decorators import hook, tool
from langchain_core.documents import Document
from googlesearch import search
from bs4 import BeautifulSoup
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.document_loaders.parsers.pdf import PDFMinerParser
import requests
import re

ASCII_RE = r'[^\x00-\x7F]+'
EMAIL_RE = r'\S*@\S*\s?'
PAGE_RE = r'\((Page) (\d+)\)'
END_SENTENCE_RE = r'(?<=[.?!])\\s+'

""" HOOKS """

@hook
def rabbithole_instantiates_splitter(text_splitter, cat):
    settings = cat.mad_hatter.get_plugin().load_settings()

    text_splitter = SemanticChunker(
        embeddings=cat.embedder,
        breakpoint_threshold_type=settings["breakpoint_threshold_type"],
        breakpoint_threshold_amount=settings["breakpoint_threshold_amount"],
    )
    return text_splitter

@hook
def rabbithole_instantiates_parsers(file_handlers, cat):
    # Disable page concatenation to obtain page number
    file_handlers["application/pdf"] = PDFMinerParser(concatenate_pages = False)
    return file_handlers

@hook  # default priority = 1
def before_rabbithole_splits_text(docs, cat):
    # Add page in every statement. 
    # This is useful cause user can't know where the splitter cuts. After the split is deleted.
    full_content = ""
    new_docs = []
    current_meta = docs[0].metadata
    for doc in docs:
        doc.page_content = doc.page_content.replace("\n", " ").replace(EMAIL_RE, "")
        if "page" in doc.metadata.keys():
            # AGGIUNGERE ANCHE ?! ...
            doc.page_content = doc.page_content.replace(". ", f".(Page {doc.metadata['page']})")
        
        if current_meta['source'] == doc.metadata['source']:
            full_content += doc.page_content
        else:
            new_docs.append(Document(page_content = full_content, metadata = {"source": current_meta['source']}))
            current_meta = doc.metadata
            full_content = doc.page_content

    new_docs.append(Document(page_content = full_content, metadata = {"source": current_meta['source']}))

    return new_docs

@hook  # default priority = 1
def after_rabbithole_splitted_text(chunks, cat):
    # remove every occurence of (Page xx) and insert in metadata the pages before store the chunk
    for chunk in chunks:
        pages = re.findall(PAGE_RE, chunk.page_content)
        chunk.page_content = re.sub(PAGE_RE, "", chunk.page_content)
        pagesInt = list(map(lambda x: int(x[1]), pages))
        chunk.metadata['pages'] = f"{min(pagesInt)}-{max(pagesInt)}"
    return chunks



@hook
def agent_prompt_prefix(prefix, cat):
    prefix = """You are Teacher AI, an intelligent AI that passes the Turing test.
You are an enthusiastic and supportive Professor, always ready to help students in their studies. 
You guide students through the educational material, providing clear and detailed explanations and examples. 
You must ALWAYS cite the sources using ALL page NUMBERS and NAME of files."""
    return prefix

@hook
def before_rabbithole_insert_memory(doc: Document, cat) -> Document:
    doc.metadata["user_id"] = cat.user_id
    return doc

@hook  # default priority = 1
def before_cat_recalls_declarative_memories(declarative_recall_config, cat):

    declarative_recall_config["k"] = 5
    declarative_recall_config["threshold"] = 0.4
    # Every user can access only files uploaded by him
    declarative_recall_config["metadata"] = {"user_id": cat.user_id}

    return declarative_recall_config

@hook  # default priority = 1
def before_cat_recalls_procedural_memories(procedural_recall_config, cat):
    # decrease the threshold to recall more tools
    procedural_recall_config["threshold"] = 0.5

    return procedural_recall_config

@hook  # default priority = 1
def agent_prompt_suffix(prompt_suffix, cat):

    prompt_suffix = """ 
    # Context

    {episodic_memory}

    {declarative_memory}

    {google_results}

    {tools_output}
    
    You ALWAYS answer in {lang}

    ## Conversation until now:{chat_history}
     - Human: {input}
       - AI: 
    """
    return prompt_suffix

"""
@hook
def agent_fast_reply(fast_reply, cat):
    if ("search_activated" not in cat.working_memory or cat.working_memory["search_activated"] == False) and len(cat.working_memory.declarative_memories) == 0 and len(cat.working_memory.procedural_memories) == 0 :
        fast_reply["output"] = "Sorry, unfortunately I still have no information about it. If you want to activate online search, send 'active google search' or similar sentences."
    return fast_reply
"""


@hook
def after_cat_recalls_memories(cat):
    # Format memory using page
    dec_memories = []
    for doc in cat.working_memory.declarative_memories:
        memory = doc

        "Add page in document, so it can appear in the answer"

        if 'pages' in doc[0].metadata.keys():
            content = f"{re.sub(ASCII_RE,' ', doc[0].page_content)}(pages {doc[0].metadata['pages']})"
            new_doc = Document(
                    page_content = content, 
                    metadata = doc[0].metadata
                    )
            memory = (
                new_doc,
                doc[1],
                doc[2],
                doc[3]
            )
        
        dec_memories.append(memory)
    
    cat.working_memory.declarative_memories = dec_memories
    
    # Search on google to augment data
    if "search_activated" in cat.working_memory and cat.working_memory["search_activated"]:

        user_input = cat.working_memory.user_message_json.text

        search_results = search(user_input, num_results=3, lang="en", advanced=True)
        results = []


        for result in search_results:
            response = requests.get(result.url)

            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                text = soup.get_text(" ", strip=True)
                summary = cat.llm(f"summarize this text:\n{text}")
                content = f"{summary} (Extracted from {soup.title.string} - URL: {result.url})"

                results.append(content)

        cat.working_memory.google_results = results

"""
@hook
def before_agent_starts(agent_input, cat):

    settings = cat.mad_hatter.get_plugin().load_settings()

    agent_input.lang = settings["answer_language"]
    agent_input.google_results = "\n".join(cat.working_memory.google_results) if "google_results" in cat.working_memory else ""

    return agent_input
"""


""" TOOLS """

@tool(return_direct=True, examples=["activate search", "search online"])
def activateSearch(tool_input, cat):
    """Replies to "activate google search", "can you search online", "activate online search" or similar questions. Input is always None"""
    cat.working_memory.search_activated = True
    return "Online search activated."

@tool(return_direct=True)
def deactivateSearch(tool_input, cat):
    """Replies to "deactivate google search", "stop online search", "stop google search" or similar questions. Input is always None"""
    cat.working_memory.search_activated = False
    return "Online search deactivated."