# Author Aimped
# Date: 2023-May-25
# Description: This file contains standart input and output (sio) keys for each task

from enum import Enum
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Union

from pydantic import BaseModel, Extra, Field

data_types = [
    'data_json',
    'data_pdf',
    'data_image',
    'data_svg',
    'data_audio',
    'data_csv',
    'data_txt',
    'data_excel',
    'data_docx',
    'data_xml',
    'data_video',
    'data_zip',
    'data_char',
    'data_file',
    'raw_output',
    'data_url'
]


################ Task: Text Classification ################

class TextClassificationInput(BaseModel):
    """Input data for text classification task"""
    text: List[str] = Field(..., description="List of texts", example=["This is an example text for classification.",
                                                                       "This is another example text for classification."])
    class Config:
        extra = Extra.allow

class TextClassificationOutput(BaseModel):
    """Output data for text classification task"""
    status: bool = True
    data_type: List[str] = Field(default=["data_json"], description="List of data types", enum=data_types)
    output: Dict[str, Any] = Field(
        default={"data_json": {"result": None}},
        description="data_json output dictionary",
        keys={"enum": data_types},
        example={
            'data_json': {
                'result': [
                    {
                        'category': [str],
                        'classes': [{'label': str, 'score': float},
                                    {'label': str, 'score': float}]
                    }
                ]
            }
        }
    )

    def __init__(self, model_prediction: Any, **data: Any):
        super().__init__(**data)
        self.output['data_json']['result'] = model_prediction


################ Task: Name Entity Recognition ################

class NameEntityRecognitionInput(BaseModel):
    """Input data for name entity recognition task"""
    text: List[str] = Field(..., description="List of texts", example=["This is an example text for NER."])
    entity: List[Any] = Field(..., description="List of entities", example=["PERSON", "ORG", "GPE"])

    class Config:
        extra = Extra.allow


class NameEntityRecognitionOutput(BaseModel):
    """Output data for name entity recognition task"""
    status: bool = True
    data_type: List[str] = Field(default=["data_json"], description="List of data types", enum=data_types)
    output: Dict[str, Any] = Field(
        default={"data_json": {"result": None}},
        description="Output dictionary",
        keys={"enum": data_types},
        example={
            'data_json': {
                'result': [
                    [{
                        'entity': str,
                        'confidence': float,
                        'chunk': str,
                        'begin': int,
                        'end': int
                    }]
                ]
            }
        }
    )

    def __init__(self, model_prediction: Any, **data: Any):
        super().__init__(**data)
        self.output['data_json']['result'] = model_prediction


######################### Task: Deidentification #########################

class DeidentificationInput(BaseModel):
    """Input data for deidentification task"""
    text: List[str] = Field(..., description="List of texts", example=["This is an example text for deidentification."])
    entity: List[str] = Field(..., description="List of entities", example=["PERSON", "ORG", "GPE"])

    class Config:
        extra = Extra.allow

class DeidentificationOutput(BaseModel):
    """Output data for deidentification task"""
    status: bool = True
    data_type: List[str] = Field(default=["data_json"], description="List of data types", enum=data_types)
    output: Dict[str, Any] = Field(
                                    default={"data_json": {"result": None}},
                                    description="Output dictionary",
                                    keys={"enum": data_types},
                                    example={
                                        'data_json': {'result': [
                                            {'entities': [
                                                {
                                                    'entity': str,
                                                    'confidence': float,
                                                    'chunk': str,
                                                    'begin': int,
                                                    'end': int,
                                                    'faked_chunk': str
                                                }],
                                                'masked_text': str,
                                                'faked_text': str
                                            }]
                                        }}
                                )

    def __init__(self, model_prediction: Any, **data: Any):
        super().__init__(**data)
        self.output['data_json']['result'] = model_prediction


####################### Task: Chatbot ###################### TODO: Still in development

class ChatbotInput(BaseModel):
    """Input data for chatbot task"""
    messages: List[Dict[str, str]] = Field(..., description="List of messages",
                                           example=[{"role": "user", "content": "Who are you?"},
                                                    {"role": "assistant", "content": "I am Chat Bot. Who are you?"},
                                                    {"role": "user", "content": "I am Joseph."}
                                                    ])
    stream: bool = False
    max_tokens: int = 256
    temperature: float = 0.7

    class Config:
        extra = Extra.allow

class ChatbotOutput(BaseModel):
    """Output data for chatbot task"""
    status: bool = True
    data_type: List[str] = Field(default=["data_json", "data_audio"],
                                 description="List of data types",
                                 enum=data_types)
    # data_json: Dict[str, Any] = Field(default=None, description="Text API Output in dictionary", repr=False, exclude=True,)
    # data_audio: str = Field(default=None, description="Audio API Output in base64 format", repr=False, exclude=True,
    #                         example="data:audio/mp3;base64,//tQxA-audio-base64-format-string", )
    raw_output: Dict[str, Any] = Field(default=None, description="Raw API Output in dictionary", repr=False,
                                       exclude=True)
    output: Dict[str, Any] = Field(
        default={"data_json": {"text": None}, "data_audio": None, "raw_output": None},
        description="Output dictionary",
        keys={"enum": data_types},
        example={
            "data_json": {"text": "Well…   good luck…"},
            "raw_output": {'choices': [{'message': {'content': 'Well… good luck…', 'role': 'assistant',
                                                    'finish_reason': 'stop', 'index': 0, }}],
                           'created': 1683618639,
                           'id': 'chaxxpl-7EXXXXXwQNqPmXXXXXaYWkMtKNVjg',
                           'model': 'gpt-3.5-turbo-0301',
                           'object': 'chat.completion',
                           'usage': {'completion_tokens': 27,
                                     'prompt_tokens': 186,
                                     'total_tokens': 213}},
            "data_audio": "data:audio/mp3;base64,//tQxA-audio-base64-format-audio-file...==="},
    )

    def __init__(self, api_text: Any, api_result: Any, api_audio: Any, **data: Any):
        super().__init__(**data)
        self.output['data_json']['text'] = api_text
        self.output['raw_output'] = api_result
        self.output['data_audio'] = api_audio
      

######################## Task: OCR ######################## TODO: Still in development

class OcrInput(BaseModel):
    """Input data for OCR task"""
    file_type: str = Field(..., description="File type image or pdf", example="image/png")
    file: str = Field(..., description="File in base64 format", example="data:image/png;base64,//tQxA-image-base64-format-string")

    class Config:
        extra = Extra.allow

class OcrOutput(BaseModel):
    """Output data for OCR task"""
    status: bool = True
    data_type: List[str] = Field(default=["data_image"], description="List of data types", enum=data_types)
    output: Dict[str, Any] = Field(
                                    default={},
                                    description="Output dictionary",
                                    keys={"enum": data_types},
                                    example={"data_image": ["data:image/png;base64,//tQxA-image-base64-format-string"]}
                                )

    def __init__(self, model_prediction: Any, data_type: List[str]=["data_image"], **data: Any):
        super().__init__(**data)
        self.data_type = data_type

        if "data_image" in self.data_type:
            self.output['data_image'] = model_prediction
        elif "data_pdf" in self.data_type:
            self.output['data_pdf'] = model_prediction
        else:
            raise ValueError("Invalid data_type, data_type must be data_image or data_pdf")
        
######################## Task: Translation ########################

class TranslationInput(BaseModel):
    """Input data for translation task"""
    text: List[str] = Field(..., description="List of texts", example=["This is an example text for translation."])
    source_language: str = Field(..., description="Source language", example="en")
    output_language: str = Field(..., description="Output language", example="de")

    class Config:
        extra = Extra.allow


class TranslationOutput(BaseModel):
    """Output data for translation task"""
    status: bool = True
    data_type: List[str] = Field(default=["data_json"], description="List of data types one or more item of data_types", enum=data_types)
    output: Dict[str, Any] = Field(
                                    default={"data_json": {"result": {'output_language':None, 'translated_text':None}}},
                                    description="Output dictionary",
                                    keys={"enum": data_types},
                                    example={
                                            "data_json": {
                                                            "result": {
                                                                        "output_language": "<output_language>",
                                                                        "translated_text": ["<translated_text>"]
                                                                    }}}
                                )

    def __init__(self, model_prediction: Any, output_language: str, **data: Any):
        super().__init__(**data)
        self.output['data_json']['result']['output_language'] = output_language
        self.output['data_json']['result']['translated_text'] = model_prediction

######################## Task: Auto Speech Recognition ########################

class ASRInput(BaseModel):
    """Input data for ASR task"""
    data_type: List[str] = Field(default=["data_audio"], description="List of data types one or more item of data_types", enum=data_types)
    file: str = Field(default=None, description="File in base64 format", example="data:audio/mp3;base64,//tQxA-audio-base64-format-audio-file...===")
    url: str = Field(default=None, description="Audio file url", example="https://www.example.com/audio.mp3")
    translation: bool = Field(default=False, description="Translation flag", example=True)
    language: str = Field(default=None, description="Language code", example="en")

    class Config:
        extra = Extra.allow

class ASROutput(BaseModel):
    status: bool = True
    data_type: List[str] = Field(default=["data_json"], description="List of data types one or more item of data_types", enum=data_types)
    output: Dict[str, Any] = Field(default={"data_json": {"transcription":None, 
                                                          "translation":None, 
                                                          "segments":None, 
                                                          "language":None}}, description="Output dictionary", example={"data_json": {"transcription":"transcription_text", "translation": "translation_to_english_text_or_None", "segments": [{"text": "text_chunk_of_transcription.", "start": 0.0, "end": 3.0}], "language": "en"}})

    def __init__(self, transcription: Any, translation=None, language:str="en", segments: Any=None, **data: Any):
        super().__init__(**data)
        self.output['data_json']['transcription'] = transcription
        self.output['data_json']['translation'] = translation
        self.output['data_json']['language'] = language
        self.output['data_json']['segments'] = segments

        class Config:
            extra = Extra.allow

