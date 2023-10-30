import os
import streamlit as st
import pytube
from pytube import YouTube
import transformers
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from youtube_transcript_api import YouTubeTranscriptApi
from huggingface_hub import InferenceClient
import librosa
import soundfile as sf
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains import ReduceDocumentsChain, MapReduceDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
import gradio as gr

# Initialize Streamlit
st.set_page_config(layout="wide")
st.title("YouTube Video Summarizer")

# Function to get YouTube video title
def get_youtube_title(url):
    yt = YouTube(url)
    return yt.title

# Function to summarize a YouTube video
def summarize_youtube_video(url, force_transcribe, api_token, temperature, words, do_sample):
    save_dir = os.path.join('./', 'docs')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    title, text, transcript_source = transcribe_youtube_video(url, force_transcribe, True, api_token)

    summary, summary_source = summarize_text(title, text, temperature, words, True, api_token, do_sample)

    return title, text, transcript_source, summary, summary_source

# Function to transcribe a YouTube video
def transcribe_youtube_video(url, force_transcribe, use_api, api_token):
    yt = YouTube(str(url))
    text = ''

    try:
        text = get_yt_transcript(url)
    except:
        pass

    if text == '' or force_transcribe:
        if use_api:
            text = transcribe_yt_vid_api(url, api_token=api_token)
            transcript_source = 'The transcript was generated using {} via the Hugging Face Hub API.'.format(transcription_model_id)
        else:
            text = transcribe_yt_vid(url)
            transcript_source = 'The transcript was generated using {} hosted locally.'.format(transcription_model_id)
    else:
        transcript_source = 'The transcript was downloaded from YouTube.'

    return yt.title, text, transcript_source

# Function to get the YouTube transcript
def get_yt_transcript(url):
    text = ''
    vid_id = pytube.extract.video_id(url)
    temp = YouTubeTranscriptApi.get_transcript(vid_id)
    for t in temp:
        text += t['text'] + ' '
    return text

# Function to transcribe a YouTube video using the Hugging Face Hub API
def transcribe_yt_vid_api(url, api_token):
    yt = YouTube(str(url))
    audio = yt.streams.filter(only_audio=True).first()
    out_file = audio.download(filename="audio.wav", output_path=save_dir)

    text = ''
    t = 25  # audio chunk length in seconds
    x, sr = librosa.load(out_file, sr=None)

    for i in range(0, (len(x) // (t * sr)) + 1):
        y = x[t * sr * i: t * sr * (i + 1)]
        split_path = os.path.join(save_dir, "audio_split.wav")
        sf.write(split_path, y, sr)
        text += client.automatic_speech_recognition(split_path)

    return text

# Function to summarize text
def summarize_text(title, text, temperature, words, use_api, api_token, do_sample):
    # ... (your existing code for summarization)
  from langchain.chains.llm import LLMChain
    from langchain.prompts import PromptTemplate
    from langchain.chains import ReduceDocumentsChain, MapReduceDocumentsChain
    from langchain.chains.combine_documents.stuff import StuffDocumentsChain
    import torch
    import transformers
    from transformers import BitsAndBytesConfig
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    from langchain import HuggingFacePipeline
    import torch

    model_kwargs1 = {"temperature":temperature ,
                    "do_sample":do_sample,
                    "min_new_tokens":200-25,
                    "max_new_tokens":200+25
                    }
    model_kwargs2 = {"temperature":temperature ,
                    "do_sample":do_sample,
                    "min_new_tokens":words-25, 
                    "max_new_tokens":words+25, 
                    'repetition_penalty':2.0
                    }
    if not do_sample:
        del model_kwargs1["temperature"]
        del model_kwargs2["temperature"]

    if use_api:
        
        from langchain import HuggingFaceHub

        # os.environ["HUGGINGFACEHUB_API_TOKEN"] = api_token
        llm=HuggingFaceHub(
            repo_id=llm_model_id, model_kwargs=model_kwargs1,
            huggingfacehub_api_token=api_token
            )
        llm2=HuggingFaceHub(
            repo_id=llm_model_id, model_kwargs=model_kwargs2,
            huggingfacehub_api_token=api_token
            )
        summary_source = 'The summary was generated using {} via Hugging Face API.'.format(llm_model_id)

    else:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            )

        tokenizer = AutoTokenizer.from_pretrained(llm_model_id)
        model = AutoModelForCausalLM.from_pretrained(llm_model_id,
                                                    # quantization_config=quantization_config
                                                    )
        model.to_bettertransformer()
        
        pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            pad_token_id=tokenizer.eos_token_id,
            **model_kwargs1,
        )
        pipeline2 = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            pad_token_id=tokenizer.eos_token_id,
            **model_kwargs2,
        )
        llm = HuggingFacePipeline(pipeline=pipeline)
        llm2 = HuggingFacePipeline(pipeline=pipeline2)

        summary_source = 'The summary was generated using {} hosted locally.'.format(llm_model_id)


    # Map
    map_template = """
    You are an intelligent AI assistant that is tasked to review the content of a video and provide a concise and accurate summary.\n
    You do not provide information that is not mentioned in the video. You only provide information that you are absolutely sure about.\n
    Summarize the following text in a clear and concise way:\n
    ----------------------- \n
    TITLE: `{title}`\n
    TEXT:\n
    `{docs}`\n
    ----------------------- \n
    BRIEF SUMMARY:\n
    """
    map_prompt = PromptTemplate(
        template = map_template, 
        input_variables = ['title','docs']
        )
    map_chain = LLMChain(llm=llm, prompt=map_prompt)

    # Reduce - Collapse
    collapse_template = """
    You are an intelligent AI assistant that is tasked to review the content of a video and provide a concise and accurate summary.\n
    You do not provide information that is not mentioned in the video. You only provide information that you are absolutely sure about.\n
    The following is set of partial summaries of a video:\n
    ----------------------- \n
    TITLE: `{title}`\n
    PARTIAL SUMMARIES:\n
    `{doc_summaries}`\n 
    ----------------------- \n
    Take these and distill them into a consolidated summary.\n
    SUMMARY:\n
    """

    collapse_prompt = PromptTemplate(
        template = collapse_template, 
        input_variables = ['title','doc_summaries']
        )
    collapse_chain = LLMChain(llm=llm, prompt=collapse_prompt)

    # Takes a list of documents, combines them into a single string, and passes this to an LLMChain
    collapse_documents_chain = StuffDocumentsChain(
        llm_chain=collapse_chain, document_variable_name="doc_summaries"
        )

    # Final Reduce - Combine
    combine_template = """\n
    You are an intelligent AI assistant that is tasked to review the content of a video and provide a concise and accurate summary.\n
    You do not provide information that is not mentioned in the video. You only provide information that you are absolutely sure about.\n
    The following is a set of partial summaries of a video:\n
    ----------------------- \n
    TITLE: `{title}`\n
    PARTIAL SUMMARIES:\n
    `{doc_summaries}`\n 
    ----------------------- \n
    Generate an executive summary of the whole text in maximum {words} words that contains the main messages, points, and arguments presented in the video as bullet points.\n
    EXECUTIVE SUMMARY:\n
    """
    combine_prompt = PromptTemplate(
        template = combine_template, 
        input_variables = ['title','doc_summaries','words']
        )
    combine_chain = LLMChain(llm=llm2, prompt=combine_prompt)

    # Takes a list of documents, combines them into a single string, and passes this to an LLMChain
    combine_documents_chain = StuffDocumentsChain(
        llm_chain=combine_chain, document_variable_name="doc_summaries"
        )

    # Combines and iteratively reduces the mapped documents
    reduce_documents_chain = ReduceDocumentsChain(
        # This is final chain that is called.
        combine_documents_chain=combine_documents_chain,
        # If documents exceed context for `StuffDocumentsChain`
        collapse_documents_chain=collapse_documents_chain,
        # The maximum number of tokens to group documents into.
        token_max=800,
        )

    # Combining documents by mapping a chain over them, then combining results
    map_reduce_chain = MapReduceDocumentsChain(
        # Map chain
        llm_chain=map_chain,
        # Reduce chain
        reduce_documents_chain=reduce_documents_chain,
        # The variable name in the llm_chain to put the documents in
        document_variable_name="docs",
        # Return the results of the map steps in the output
        return_intermediate_steps=False,
        )

    from langchain.document_loaders import TextLoader
    from langchain.text_splitter import TokenTextSplitter
    
    with open(save_dir+'/transcript.txt','w') as f:
        f.write(text)
    loader = TextLoader(save_dir+"/transcript.txt")
    doc = loader.load()
    text_splitter = TokenTextSplitter(chunk_size=800, chunk_overlap=100)
    docs = text_splitter.split_documents(doc)

    summary = map_reduce_chain.run({'input_documents':docs, 'title':title, 'words':words})

    try:
        del(map_reduce_chain,reduce_documents_chain,combine_chain,collapse_documents_chain,map_chain,collapse_chain,llm,llm2,pipeline,pipeline2,model,tokenizer)
    except:
        pass
    torch.cuda.empty_cache()

    return summary, summary_source

import gradio as gr 
import pytube
from pytube import YouTube


# Streamlit app layout
url = st.text_input("Enter YouTube video URL:", "")
api_token = st.text_input("Paste your Hugging Face API token here (Optional):", "")
force_transcribe = st.checkbox("Transcribe even if transcription is available.")
temperature = st.slider("Generation temperature", min_value=0.01, max_value=1.0, value=0.25)
words = st.slider("Length of the summary", min_value=100, max_value=500, value=100)
do_sample = st.checkbox("Set the Temperature")

if st.button("Summarize!"):
    if not url:
        st.error("Please enter a valid YouTube URL.")
    else:
        st.write("Processing... This may take a moment.")
        title, text, transcript_source = transcribe_youtube_video(url, force_transcribe, True, api_token)
        summary, summary_source = summarize_text(title, text, temperature, words, True, api_token, do_sample)

        st.write("Video Title:", title)
        st.write("Transcript Source:", transcript_source)
        st.write("Summary Source:", summary_source)
        st.write("Summary:", summary)

        # You can add other display elements as needed

