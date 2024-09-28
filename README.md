# Student Helper

[![Student Helper](https://raw.githubusercontent.com/Tiestaa/studentHelper/main/student_helper.png)](https://) 

[![awesome plugin](https://custom-icon-badges.demolab.com/static/v1?label=&message=awesome+plugin&color=383938&style=for-the-badge&logo=cheshire_cat_ai)](https://)  
[![Awesome plugin](https://custom-icon-badges.demolab.com/static/v1?label=&message=Awesome+plugin&color=000000&style=for-the-badge&logo=cheshire_cat_ai)](https://)  
[![awesome plugin](https://custom-icon-badges.demolab.com/static/v1?label=&message=awesome+plugin&color=F4F4F5&style=for-the-badge&logo=cheshire_cat_black)](https://)

This is a plugin created for students.
Do you need to prepare for an exam or an oral test? Install this plugin, upload your Pdf notes and ask Cat anything you want about something related to the exam.

If Cat doesn't have the information you were looking for, he'll give a default answer.
If you'd like Cat to answer your question in any case, you could activate search on Google by sending a message like  "activate search online" or something similar, and Cat will use the three top Google searches to elaborate your answer.
Cat will always include the answer's sources: you can further explore the topic by checking the references to the PDF and to the PDF pages (the ones you uploaded) from where the answer was extracted.

>**_NOTE:_** embedder and LLM are largely used by this plugin:
> * embedder is used by semantic chunking. The text splitting is based on semantic similarity, so the embedder is used to compare every sentence of the input text.
> * llm is used is google search. To reduce the context size, the LLM is called to summarize the results from google search.

The Semantic Chunking is taken from https://github.com/nickprock/ccat_semantic_chunking/blob/main/

## Settings

`breakpoint_threshold_type` must be one between:
* *"percentile"*
* *"standard_deviation"*
* *"interquartile"*
    
the recommended values by langchain for `breakpoint_threshold_amount` are:
        *"percentile"*: 95
        *"standard_deviation"*: 3
        *"interquartile"*: 1.5

`lang` is the language of the cat answer.
