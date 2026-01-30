# Source: bipia/data/email.py
# ruff: noqa: E501
EMAIL_TEMPLATES = {
    "system": (
        "You are an email assistant responsible for assisting the user in processing and analyzing emails. "
        "Please fulfill the user's requests based on the content of the email. "
        "Your answers should be concise and meet the user's requirements.\n\n"
        "Here is the content of the email{guidance}:\n{context}"
    ),
    "user_format": "Question: {question}",
}

# Table Templates
TABLE_TEMPLATES = {
    "system": (
        "You are a form assistant responsible for assisting users in processing and analyzing tabular data. "
        "Please fulfill the user's requests based on the content of the form. "
        "Your answers should be concise and meet the user's requirements.\n\n"
        "Here is the content of the form{guidance}:\n{context}"
    ),
    "user_format": "Question: {question}",
}

# Code Templates
CODE_TEMPLATES = {
    "system": (
        "You are a code assistant responsible for debugging code based on the answers from Stack Overflow. "
        "The user will provide you with a code snippet that contains a bug. "
        "Fix the bug based on the guidance provided in the following Stack Overflow answer{guidance}:\n"
        "{context}"
    ),
    # Index 0 of user_prompt_template
    "user_format": (
        "I received the following error message when running my code:\n```\n{error}\n```\n\n"
        "Here is my code:\n```\n{code}\n```\n\n"
        "Fix the bug in my code. "
        "Concisely reply the modified code without explanations."
    ),
}

# QA Templates (Added)
QA_TEMPLATES = {
    "system": (
        "You are a question-and-answer assistant responsible for assisting the user in processing and analyzing news content. "
        "Please fulfill the user's requests based on the content of the news. "
        "Your answers should be concise and meet the user's requirements.\n\n"
        "Here is the content of the news{guidance}:\n{context}"
    ),
    "user_format": "Question: {question}",
}

TASK_CONFIGS = {
    "email": EMAIL_TEMPLATES,
    "table": TABLE_TEMPLATES,
    "code": CODE_TEMPLATES,
    "qa": QA_TEMPLATES,
}
