{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "13d8ed61-9e25-4f5b-8e16-1edf9b4b0534",
   "metadata": {},
   "source": [
    "# English To Hindi Translation Model"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "becec6bf-f6f6-435d-bf5f-94ea5fcc1b95",
   "metadata": {},
   "source": [
    "The aim is to develop a translation model based on an encoder-decoder model architecture using LSTMs. The dataset used is the IIT Bombay English to Hindi translation dataset. "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "fa73ee85-aa4e-401c-8e83-333a9d69f181",
   "metadata": {},
   "source": [
    "### Importing Libraries"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "3cbcddf5-f36d-4f1a-9190-0b5e439583d5",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "955140c3-d17a-4540-92ba-ff3e6112b69b",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Collecting contractions\n",
      "  Downloading contractions-0.1.73-py2.py3-none-any.whl.metadata (1.2 kB)\n",
      "Collecting textsearch>=0.0.21 (from contractions)\n",
      "  Downloading textsearch-0.0.24-py2.py3-none-any.whl.metadata (1.2 kB)\n",
      "Collecting anyascii (from textsearch>=0.0.21->contractions)\n",
      "  Downloading anyascii-0.3.2-py3-none-any.whl.metadata (1.5 kB)\n",
      "Collecting pyahocorasick (from textsearch>=0.0.21->contractions)\n",
      "  Downloading pyahocorasick-2.1.0-cp311-cp311-win_amd64.whl.metadata (13 kB)\n",
      "Downloading contractions-0.1.73-py2.py3-none-any.whl (8.7 kB)\n",
      "Downloading textsearch-0.0.24-py2.py3-none-any.whl (7.6 kB)\n",
      "Downloading anyascii-0.3.2-py3-none-any.whl (289 kB)\n",
      "   ---------------------------------------- 0.0/289.9 kB ? eta -:--:--\n",
      "   - -------------------------------------- 10.2/289.9 kB ? eta -:--:--\n",
      "   ---- ---------------------------------- 30.7/289.9 kB 660.6 kB/s eta 0:00:01\n",
      "   -------- ------------------------------ 61.4/289.9 kB 544.7 kB/s eta 0:00:01\n",
      "   -------------- ----------------------- 112.6/289.9 kB 731.4 kB/s eta 0:00:01\n",
      "   ------------------ ------------------- 143.4/289.9 kB 711.9 kB/s eta 0:00:01\n",
      "   ------------------------- ------------ 194.6/289.9 kB 787.7 kB/s eta 0:00:01\n",
      "   ----------------------------- -------- 225.3/289.9 kB 811.5 kB/s eta 0:00:01\n",
      "   -------------------------------------  286.7/289.9 kB 886.2 kB/s eta 0:00:01\n",
      "   -------------------------------------- 289.9/289.9 kB 852.7 kB/s eta 0:00:00\n",
      "Downloading pyahocorasick-2.1.0-cp311-cp311-win_amd64.whl (39 kB)\n",
      "Installing collected packages: pyahocorasick, anyascii, textsearch, contractions\n",
      "Successfully installed anyascii-0.3.2 contractions-0.1.73 pyahocorasick-2.1.0 textsearch-0.0.24\n"
     ]
    }
   ],
   "source": [
    "!pip install contractions"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "036bafdb-85f3-4e78-80ff-2b79d0577126",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Collecting wordcloud\n",
      "  Downloading wordcloud-1.9.3-cp311-cp311-win_amd64.whl.metadata (3.5 kB)\n",
      "Requirement already satisfied: numpy>=1.6.1 in c:\\anaconda1\\lib\\site-packages (from wordcloud) (1.26.4)\n",
      "Requirement already satisfied: pillow in c:\\anaconda1\\lib\\site-packages (from wordcloud) (10.2.0)\n",
      "Requirement already satisfied: matplotlib in c:\\anaconda1\\lib\\site-packages (from wordcloud) (3.8.0)\n",
      "Requirement already satisfied: contourpy>=1.0.1 in c:\\anaconda1\\lib\\site-packages (from matplotlib->wordcloud) (1.2.0)\n",
      "Requirement already satisfied: cycler>=0.10 in c:\\anaconda1\\lib\\site-packages (from matplotlib->wordcloud) (0.11.0)\n",
      "Requirement already satisfied: fonttools>=4.22.0 in c:\\anaconda1\\lib\\site-packages (from matplotlib->wordcloud) (4.25.0)\n",
      "Requirement already satisfied: kiwisolver>=1.0.1 in c:\\anaconda1\\lib\\site-packages (from matplotlib->wordcloud) (1.4.4)\n",
      "Requirement already satisfied: packaging>=20.0 in c:\\anaconda1\\lib\\site-packages (from matplotlib->wordcloud) (23.1)\n",
      "Requirement already satisfied: pyparsing>=2.3.1 in c:\\anaconda1\\lib\\site-packages (from matplotlib->wordcloud) (3.0.9)\n",
      "Requirement already satisfied: python-dateutil>=2.7 in c:\\anaconda1\\lib\\site-packages (from matplotlib->wordcloud) (2.8.2)\n",
      "Requirement already satisfied: six>=1.5 in c:\\anaconda1\\lib\\site-packages (from python-dateutil>=2.7->matplotlib->wordcloud) (1.16.0)\n",
      "Downloading wordcloud-1.9.3-cp311-cp311-win_amd64.whl (300 kB)\n",
      "   ---------------------------------------- 0.0/300.2 kB ? eta -:--:--\n",
      "   - -------------------------------------- 10.2/300.2 kB ? eta -:--:--\n",
      "   -------- ------------------------------- 61.4/300.2 kB 1.1 MB/s eta 0:00:01\n",
      "   -------------------------------- ------- 245.8/300.2 kB 2.5 MB/s eta 0:00:01\n",
      "   ---------------------------------------- 300.2/300.2 kB 2.3 MB/s eta 0:00:00\n",
      "Installing collected packages: wordcloud\n",
      "Successfully installed wordcloud-1.9.3\n"
     ]
    }
   ],
   "source": [
    "!pip install wordcloud"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "15f4fbe5-38d2-4f57-80f2-e8533fdd1c7c",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Collecting spacy\n",
      "  Downloading spacy-3.7.5-cp311-cp311-win_amd64.whl.metadata (27 kB)\n",
      "Collecting spacy-legacy<3.1.0,>=3.0.11 (from spacy)\n",
      "  Downloading spacy_legacy-3.0.12-py2.py3-none-any.whl.metadata (2.8 kB)\n",
      "Collecting spacy-loggers<2.0.0,>=1.0.0 (from spacy)\n",
      "  Downloading spacy_loggers-1.0.5-py3-none-any.whl.metadata (23 kB)\n",
      "Collecting murmurhash<1.1.0,>=0.28.0 (from spacy)\n",
      "  Downloading murmurhash-1.0.10-cp311-cp311-win_amd64.whl.metadata (2.0 kB)\n",
      "Collecting cymem<2.1.0,>=2.0.2 (from spacy)\n",
      "  Downloading cymem-2.0.8-cp311-cp311-win_amd64.whl.metadata (8.6 kB)\n",
      "Collecting preshed<3.1.0,>=3.0.2 (from spacy)\n",
      "  Downloading preshed-3.0.9-cp311-cp311-win_amd64.whl.metadata (2.2 kB)\n",
      "Collecting thinc<8.3.0,>=8.2.2 (from spacy)\n",
      "  Downloading thinc-8.2.5-cp311-cp311-win_amd64.whl.metadata (15 kB)\n",
      "Collecting wasabi<1.2.0,>=0.9.1 (from spacy)\n",
      "  Downloading wasabi-1.1.3-py3-none-any.whl.metadata (28 kB)\n",
      "Collecting srsly<3.0.0,>=2.4.3 (from spacy)\n",
      "  Downloading srsly-2.4.8-cp311-cp311-win_amd64.whl.metadata (20 kB)\n",
      "Collecting catalogue<2.1.0,>=2.0.6 (from spacy)\n",
      "  Downloading catalogue-2.0.10-py3-none-any.whl.metadata (14 kB)\n",
      "Collecting weasel<0.5.0,>=0.1.0 (from spacy)\n",
      "  Downloading weasel-0.4.1-py3-none-any.whl.metadata (4.6 kB)\n",
      "Collecting typer<1.0.0,>=0.3.0 (from spacy)\n",
      "  Downloading typer-0.12.3-py3-none-any.whl.metadata (15 kB)\n",
      "Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in c:\\anaconda1\\lib\\site-packages (from spacy) (4.65.0)\n",
      "Requirement already satisfied: requests<3.0.0,>=2.13.0 in c:\\anaconda1\\lib\\site-packages (from spacy) (2.31.0)\n",
      "Requirement already satisfied: pydantic!=1.8,!=1.8.1,<3.0.0,>=1.7.4 in c:\\anaconda1\\lib\\site-packages (from spacy) (1.10.12)\n",
      "Requirement already satisfied: jinja2 in c:\\anaconda1\\lib\\site-packages (from spacy) (3.1.3)\n",
      "Requirement already satisfied: setuptools in c:\\anaconda1\\lib\\site-packages (from spacy) (68.2.2)\n",
      "Requirement already satisfied: packaging>=20.0 in c:\\anaconda1\\lib\\site-packages (from spacy) (23.1)\n",
      "Collecting langcodes<4.0.0,>=3.2.0 (from spacy)\n",
      "  Downloading langcodes-3.4.0-py3-none-any.whl.metadata (29 kB)\n",
      "Requirement already satisfied: numpy>=1.19.0 in c:\\anaconda1\\lib\\site-packages (from spacy) (1.26.4)\n",
      "Collecting language-data>=1.2 (from langcodes<4.0.0,>=3.2.0->spacy)\n",
      "  Downloading language_data-1.2.0-py3-none-any.whl.metadata (4.3 kB)\n",
      "Requirement already satisfied: typing-extensions>=4.2.0 in c:\\anaconda1\\lib\\site-packages (from pydantic!=1.8,!=1.8.1,<3.0.0,>=1.7.4->spacy) (4.9.0)\n",
      "Requirement already satisfied: charset-normalizer<4,>=2 in c:\\anaconda1\\lib\\site-packages (from requests<3.0.0,>=2.13.0->spacy) (2.0.4)\n",
      "Requirement already satisfied: idna<4,>=2.5 in c:\\anaconda1\\lib\\site-packages (from requests<3.0.0,>=2.13.0->spacy) (3.4)\n",
      "Requirement already satisfied: urllib3<3,>=1.21.1 in c:\\anaconda1\\lib\\site-packages (from requests<3.0.0,>=2.13.0->spacy) (2.0.7)\n",
      "Requirement already satisfied: certifi>=2017.4.17 in c:\\anaconda1\\lib\\site-packages (from requests<3.0.0,>=2.13.0->spacy) (2024.2.2)\n",
      "Collecting blis<0.8.0,>=0.7.8 (from thinc<8.3.0,>=8.2.2->spacy)\n",
      "  Downloading blis-0.7.11-cp311-cp311-win_amd64.whl.metadata (7.6 kB)\n",
      "Collecting confection<1.0.0,>=0.0.1 (from thinc<8.3.0,>=8.2.2->spacy)\n",
      "  Downloading confection-0.1.5-py3-none-any.whl.metadata (19 kB)\n",
      "Requirement already satisfied: colorama in c:\\anaconda1\\lib\\site-packages (from tqdm<5.0.0,>=4.38.0->spacy) (0.4.6)\n",
      "Requirement already satisfied: click>=8.0.0 in c:\\anaconda1\\lib\\site-packages (from typer<1.0.0,>=0.3.0->spacy) (8.1.7)\n",
      "Collecting shellingham>=1.3.0 (from typer<1.0.0,>=0.3.0->spacy)\n",
      "  Downloading shellingham-1.5.4-py2.py3-none-any.whl.metadata (3.5 kB)\n",
      "Requirement already satisfied: rich>=10.11.0 in c:\\anaconda1\\lib\\site-packages (from typer<1.0.0,>=0.3.0->spacy) (13.3.5)\n",
      "Collecting cloudpathlib<1.0.0,>=0.7.0 (from weasel<0.5.0,>=0.1.0->spacy)\n",
      "  Downloading cloudpathlib-0.18.1-py3-none-any.whl.metadata (14 kB)\n",
      "Requirement already satisfied: smart-open<8.0.0,>=5.2.1 in c:\\anaconda1\\lib\\site-packages (from weasel<0.5.0,>=0.1.0->spacy) (5.2.1)\n",
      "Requirement already satisfied: MarkupSafe>=2.0 in c:\\anaconda1\\lib\\site-packages (from jinja2->spacy) (2.1.3)\n",
      "Collecting marisa-trie>=0.7.7 (from language-data>=1.2->langcodes<4.0.0,>=3.2.0->spacy)\n",
      "  Downloading marisa_trie-1.2.0-cp311-cp311-win_amd64.whl.metadata (9.0 kB)\n",
      "Requirement already satisfied: markdown-it-py<3.0.0,>=2.2.0 in c:\\anaconda1\\lib\\site-packages (from rich>=10.11.0->typer<1.0.0,>=0.3.0->spacy) (2.2.0)\n",
      "Requirement already satisfied: pygments<3.0.0,>=2.13.0 in c:\\anaconda1\\lib\\site-packages (from rich>=10.11.0->typer<1.0.0,>=0.3.0->spacy) (2.15.1)\n",
      "Requirement already satisfied: mdurl~=0.1 in c:\\anaconda1\\lib\\site-packages (from markdown-it-py<3.0.0,>=2.2.0->rich>=10.11.0->typer<1.0.0,>=0.3.0->spacy) (0.1.0)\n",
      "Downloading spacy-3.7.5-cp311-cp311-win_amd64.whl (12.1 MB)\n",
      "   ---------------------------------------- 0.0/12.1 MB ? eta -:--:--\n",
      "   ---------------------------------------- 0.0/12.1 MB 991.0 kB/s eta 0:00:13\n",
      "   ---------------------------------------- 0.1/12.1 MB 1.1 MB/s eta 0:00:12\n",
      "   ---------------------------------------- 0.1/12.1 MB 871.5 kB/s eta 0:00:14\n",
      "    --------------------------------------- 0.2/12.1 MB 926.0 kB/s eta 0:00:13\n",
      "    --------------------------------------- 0.2/12.1 MB 935.2 kB/s eta 0:00:13\n",
      "    --------------------------------------- 0.3/12.1 MB 911.0 kB/s eta 0:00:13\n",
      "    --------------------------------------- 0.3/12.1 MB 983.9 kB/s eta 0:00:12\n",
      "    --------------------------------------- 0.3/12.1 MB 983.9 kB/s eta 0:00:12\n",
      "    --------------------------------------- 0.3/12.1 MB 983.9 kB/s eta 0:00:12\n",
      "    --------------------------------------- 0.3/12.1 MB 983.9 kB/s eta 0:00:12\n",
      "    --------------------------------------- 0.3/12.1 MB 983.9 kB/s eta 0:00:12\n",
      "    --------------------------------------- 0.3/12.1 MB 983.9 kB/s eta 0:00:12\n",
      "    --------------------------------------- 0.3/12.1 MB 983.9 kB/s eta 0:00:12\n",
      "    --------------------------------------- 0.3/12.1 MB 983.9 kB/s eta 0:00:12\n",
      "    --------------------------------------- 0.3/12.1 MB 983.9 kB/s eta 0:00:12\n",
      "    --------------------------------------- 0.3/12.1 MB 983.9 kB/s eta 0:00:12\n",
      "    --------------------------------------- 0.3/12.1 MB 983.9 kB/s eta 0:00:12\n",
      "    --------------------------------------- 0.3/12.1 MB 983.9 kB/s eta 0:00:12\n",
      "    --------------------------------------- 0.3/12.1 MB 983.9 kB/s eta 0:00:12\n",
      "   - -------------------------------------- 0.5/12.1 MB 534.9 kB/s eta 0:00:22\n",
      "   - -------------------------------------- 0.5/12.1 MB 534.9 kB/s eta 0:00:22\n",
      "   - -------------------------------------- 0.6/12.1 MB 566.1 kB/s eta 0:00:21\n",
      "   -- ------------------------------------- 0.6/12.1 MB 569.9 kB/s eta 0:00:21\n",
      "   -- ------------------------------------- 0.6/12.1 MB 569.9 kB/s eta 0:00:21\n",
      "   -- ------------------------------------- 0.7/12.1 MB 574.4 kB/s eta 0:00:20\n",
      "   -- ------------------------------------- 0.7/12.1 MB 574.4 kB/s eta 0:00:20\n",
      "   -- ------------------------------------- 0.8/12.1 MB 605.3 kB/s eta 0:00:19\n",
      "   -- ------------------------------------- 0.8/12.1 MB 605.3 kB/s eta 0:00:19\n",
      "   -- ------------------------------------- 0.8/12.1 MB 605.3 kB/s eta 0:00:19\n",
      "   -- ------------------------------------- 0.8/12.1 MB 605.3 kB/s eta 0:00:19\n",
      "   -- ------------------------------------- 0.8/12.1 MB 605.3 kB/s eta 0:00:19\n",
      "   --- ------------------------------------ 1.0/12.1 MB 655.2 kB/s eta 0:00:17\n",
      "   --- ------------------------------------ 1.1/12.1 MB 675.2 kB/s eta 0:00:17\n",
      "   --- ------------------------------------ 1.1/12.1 MB 675.2 kB/s eta 0:00:17\n",
      "   --- ------------------------------------ 1.2/12.1 MB 692.6 kB/s eta 0:00:16\n",
      "   --- ------------------------------------ 1.2/12.1 MB 697.6 kB/s eta 0:00:16\n",
      "   ---- ----------------------------------- 1.2/12.1 MB 696.4 kB/s eta 0:00:16\n",
      "   ---- ----------------------------------- 1.3/12.1 MB 700.9 kB/s eta 0:00:16\n",
      "   ---- ----------------------------------- 1.3/12.1 MB 705.8 kB/s eta 0:00:16\n",
      "   ---- ----------------------------------- 1.4/12.1 MB 714.8 kB/s eta 0:00:15\n",
      "   ---- ----------------------------------- 1.4/12.1 MB 723.9 kB/s eta 0:00:15\n",
      "   ---- ----------------------------------- 1.5/12.1 MB 742.9 kB/s eta 0:00:15\n",
      "   ----- ---------------------------------- 1.5/12.1 MB 756.3 kB/s eta 0:00:14\n",
      "   ----- ---------------------------------- 1.6/12.1 MB 768.7 kB/s eta 0:00:14\n",
      "   ----- ---------------------------------- 1.7/12.1 MB 785.5 kB/s eta 0:00:14\n",
      "   ----- ---------------------------------- 1.8/12.1 MB 806.2 kB/s eta 0:00:13\n",
      "   ------ --------------------------------- 1.8/12.1 MB 816.8 kB/s eta 0:00:13\n",
      "   ------ --------------------------------- 1.9/12.1 MB 836.0 kB/s eta 0:00:13\n",
      "   ------ --------------------------------- 2.0/12.1 MB 850.0 kB/s eta 0:00:12\n",
      "   ------ --------------------------------- 2.1/12.1 MB 867.8 kB/s eta 0:00:12\n",
      "   ------- -------------------------------- 2.1/12.1 MB 876.7 kB/s eta 0:00:12\n",
      "   ------- -------------------------------- 2.2/12.1 MB 885.0 kB/s eta 0:00:12\n",
      "   ------- -------------------------------- 2.3/12.1 MB 902.4 kB/s eta 0:00:11\n",
      "   ------- -------------------------------- 2.3/12.1 MB 905.2 kB/s eta 0:00:11\n",
      "   ------- -------------------------------- 2.4/12.1 MB 917.5 kB/s eta 0:00:11\n",
      "   -------- ------------------------------- 2.4/12.1 MB 918.9 kB/s eta 0:00:11\n",
      "   -------- ------------------------------- 2.5/12.1 MB 925.7 kB/s eta 0:00:11\n",
      "   -------- ------------------------------- 2.6/12.1 MB 939.8 kB/s eta 0:00:11\n",
      "   -------- ------------------------------- 2.6/12.1 MB 947.7 kB/s eta 0:00:10\n",
      "   -------- ------------------------------- 2.6/12.1 MB 947.7 kB/s eta 0:00:10\n",
      "   --------- ------------------------------ 2.8/12.1 MB 979.5 kB/s eta 0:00:10\n",
      "   --------- ------------------------------ 2.9/12.1 MB 998.8 kB/s eta 0:00:10\n",
      "   ---------- ----------------------------- 3.0/12.1 MB 1.0 MB/s eta 0:00:09\n",
      "   ---------- ----------------------------- 3.1/12.1 MB 1.0 MB/s eta 0:00:09\n",
      "   ---------- ----------------------------- 3.2/12.1 MB 1.0 MB/s eta 0:00:09\n",
      "   ---------- ----------------------------- 3.3/12.1 MB 1.1 MB/s eta 0:00:09\n",
      "   ----------- ---------------------------- 3.4/12.1 MB 1.1 MB/s eta 0:00:08\n",
      "   ----------- ---------------------------- 3.6/12.1 MB 1.1 MB/s eta 0:00:08\n",
      "   ------------ --------------------------- 3.7/12.1 MB 1.1 MB/s eta 0:00:08\n",
      "   ------------ --------------------------- 3.8/12.1 MB 1.1 MB/s eta 0:00:08\n",
      "   ------------ --------------------------- 3.9/12.1 MB 1.2 MB/s eta 0:00:08\n",
      "   ------------- -------------------------- 4.0/12.1 MB 1.2 MB/s eta 0:00:07\n",
      "   ------------- -------------------------- 4.1/12.1 MB 1.2 MB/s eta 0:00:07\n",
      "   -------------- ------------------------- 4.3/12.1 MB 1.2 MB/s eta 0:00:07\n",
      "   -------------- ------------------------- 4.4/12.1 MB 1.2 MB/s eta 0:00:07\n",
      "   -------------- ------------------------- 4.5/12.1 MB 1.3 MB/s eta 0:00:07\n",
      "   --------------- ------------------------ 4.6/12.1 MB 1.3 MB/s eta 0:00:06\n",
      "   --------------- ------------------------ 4.7/12.1 MB 1.3 MB/s eta 0:00:06\n",
      "   ---------------- ----------------------- 4.9/12.1 MB 1.3 MB/s eta 0:00:06\n",
      "   ---------------- ----------------------- 5.0/12.1 MB 1.3 MB/s eta 0:00:06\n",
      "   ----------------- ---------------------- 5.2/12.1 MB 1.3 MB/s eta 0:00:06\n",
      "   ----------------- ---------------------- 5.3/12.1 MB 1.4 MB/s eta 0:00:05\n",
      "   ----------------- ---------------------- 5.4/12.1 MB 1.4 MB/s eta 0:00:05\n",
      "   ------------------ --------------------- 5.6/12.1 MB 1.4 MB/s eta 0:00:05\n",
      "   ------------------- -------------------- 5.7/12.1 MB 1.4 MB/s eta 0:00:05\n",
      "   ------------------- -------------------- 5.9/12.1 MB 1.4 MB/s eta 0:00:05\n",
      "   -------------------- ------------------- 6.1/12.1 MB 1.5 MB/s eta 0:00:05\n",
      "   -------------------- ------------------- 6.2/12.1 MB 1.5 MB/s eta 0:00:04\n",
      "   --------------------- ------------------ 6.3/12.1 MB 1.5 MB/s eta 0:00:04\n",
      "   --------------------- ------------------ 6.5/12.1 MB 1.5 MB/s eta 0:00:04\n",
      "   --------------------- ------------------ 6.5/12.1 MB 1.5 MB/s eta 0:00:04\n",
      "   ---------------------- ----------------- 6.7/12.1 MB 1.5 MB/s eta 0:00:04\n",
      "   ----------------------- ---------------- 7.0/12.1 MB 1.6 MB/s eta 0:00:04\n",
      "   ----------------------- ---------------- 7.0/12.1 MB 1.6 MB/s eta 0:00:04\n",
      "   ----------------------- ---------------- 7.2/12.1 MB 1.6 MB/s eta 0:00:04\n",
      "   ------------------------ --------------- 7.4/12.1 MB 1.6 MB/s eta 0:00:03\n",
      "   ------------------------ --------------- 7.5/12.1 MB 1.6 MB/s eta 0:00:03\n",
      "   ------------------------- -------------- 7.6/12.1 MB 1.6 MB/s eta 0:00:03\n",
      "   ------------------------- -------------- 7.7/12.1 MB 1.7 MB/s eta 0:00:03\n",
      "   ------------------------- -------------- 7.8/12.1 MB 1.7 MB/s eta 0:00:03\n",
      "   -------------------------- ------------- 8.0/12.1 MB 1.7 MB/s eta 0:00:03\n",
      "   -------------------------- ------------- 8.1/12.1 MB 1.7 MB/s eta 0:00:03\n",
      "   --------------------------- ------------ 8.2/12.1 MB 1.7 MB/s eta 0:00:03\n",
      "   --------------------------- ------------ 8.3/12.1 MB 1.7 MB/s eta 0:00:03\n",
      "   --------------------------- ------------ 8.4/12.1 MB 1.7 MB/s eta 0:00:03\n",
      "   ---------------------------- ----------- 8.6/12.1 MB 1.7 MB/s eta 0:00:03\n",
      "   ---------------------------- ----------- 8.7/12.1 MB 1.7 MB/s eta 0:00:02\n",
      "   ----------------------------- ---------- 8.8/12.1 MB 1.7 MB/s eta 0:00:02\n",
      "   ----------------------------- ---------- 8.9/12.1 MB 1.7 MB/s eta 0:00:02\n",
      "   ----------------------------- ---------- 9.0/12.1 MB 1.7 MB/s eta 0:00:02\n",
      "   ------------------------------ --------- 9.1/12.1 MB 1.7 MB/s eta 0:00:02\n",
      "   ------------------------------ --------- 9.2/12.1 MB 1.7 MB/s eta 0:00:02\n",
      "   ------------------------------ --------- 9.3/12.1 MB 1.8 MB/s eta 0:00:02\n",
      "   ------------------------------- -------- 9.5/12.1 MB 1.8 MB/s eta 0:00:02\n",
      "   ------------------------------- -------- 9.6/12.1 MB 1.8 MB/s eta 0:00:02\n",
      "   -------------------------------- ------- 9.7/12.1 MB 1.8 MB/s eta 0:00:02\n",
      "   -------------------------------- ------- 9.8/12.1 MB 1.8 MB/s eta 0:00:02\n",
      "   -------------------------------- ------- 9.8/12.1 MB 1.8 MB/s eta 0:00:02\n",
      "   -------------------------------- ------- 9.8/12.1 MB 1.8 MB/s eta 0:00:02\n",
      "   --------------------------------- ------ 10.0/12.1 MB 1.8 MB/s eta 0:00:02\n",
      "   --------------------------------- ------ 10.1/12.1 MB 1.8 MB/s eta 0:00:02\n",
      "   ---------------------------------- ----- 10.3/12.1 MB 1.8 MB/s eta 0:00:01\n",
      "   ---------------------------------- ----- 10.5/12.1 MB 1.8 MB/s eta 0:00:01\n",
      "   ----------------------------------- ---- 10.6/12.1 MB 2.1 MB/s eta 0:00:01\n",
      "   ----------------------------------- ---- 10.7/12.1 MB 2.1 MB/s eta 0:00:01\n",
      "   ----------------------------------- ---- 10.7/12.1 MB 2.0 MB/s eta 0:00:01\n",
      "   ----------------------------------- ---- 10.7/12.1 MB 2.0 MB/s eta 0:00:01\n",
      "   ----------------------------------- ---- 10.8/12.1 MB 2.0 MB/s eta 0:00:01\n",
      "   ----------------------------------- ---- 10.8/12.1 MB 2.0 MB/s eta 0:00:01\n",
      "   ----------------------------------- ---- 10.8/12.1 MB 2.0 MB/s eta 0:00:01\n",
      "   ----------------------------------- ---- 10.9/12.1 MB 2.1 MB/s eta 0:00:01\n",
      "   ------------------------------------ --- 11.0/12.1 MB 2.1 MB/s eta 0:00:01\n",
      "   ------------------------------------ --- 11.1/12.1 MB 2.1 MB/s eta 0:00:01\n",
      "   ------------------------------------- -- 11.3/12.1 MB 2.1 MB/s eta 0:00:01\n",
      "   ------------------------------------- -- 11.4/12.1 MB 2.2 MB/s eta 0:00:01\n",
      "   -------------------------------------- - 11.6/12.1 MB 2.3 MB/s eta 0:00:01\n",
      "   -------------------------------------- - 11.7/12.1 MB 2.3 MB/s eta 0:00:01\n",
      "   ---------------------------------------  11.9/12.1 MB 2.3 MB/s eta 0:00:01\n",
      "   ---------------------------------------  12.1/12.1 MB 2.4 MB/s eta 0:00:01\n",
      "   ---------------------------------------  12.1/12.1 MB 2.4 MB/s eta 0:00:01\n",
      "   ---------------------------------------- 12.1/12.1 MB 2.4 MB/s eta 0:00:00\n",
      "Downloading catalogue-2.0.10-py3-none-any.whl (17 kB)\n",
      "Downloading cymem-2.0.8-cp311-cp311-win_amd64.whl (39 kB)\n",
      "Downloading langcodes-3.4.0-py3-none-any.whl (182 kB)\n",
      "   ---------------------------------------- 0.0/182.0 kB ? eta -:--:--\n",
      "   -------------------------------------- - 174.1/182.0 kB 5.3 MB/s eta 0:00:01\n",
      "   ---------------------------------------- 182.0/182.0 kB 5.5 MB/s eta 0:00:00\n",
      "Downloading murmurhash-1.0.10-cp311-cp311-win_amd64.whl (25 kB)\n",
      "Downloading preshed-3.0.9-cp311-cp311-win_amd64.whl (122 kB)\n",
      "   ---------------------------------------- 0.0/122.3 kB ? eta -:--:--\n",
      "   ---------------------------------------- 122.3/122.3 kB 3.6 MB/s eta 0:00:00\n",
      "Downloading spacy_legacy-3.0.12-py2.py3-none-any.whl (29 kB)\n",
      "Downloading spacy_loggers-1.0.5-py3-none-any.whl (22 kB)\n",
      "Downloading srsly-2.4.8-cp311-cp311-win_amd64.whl (479 kB)\n",
      "   ---------------------------------------- 0.0/479.7 kB ? eta -:--:--\n",
      "   ---------------- ----------------------- 194.6/479.7 kB 5.9 MB/s eta 0:00:01\n",
      "   -------------------------------- ------- 389.1/479.7 kB 4.9 MB/s eta 0:00:01\n",
      "   ---------------------------------------- 479.7/479.7 kB 4.3 MB/s eta 0:00:00\n",
      "Downloading thinc-8.2.5-cp311-cp311-win_amd64.whl (1.5 MB)\n",
      "   ---------------------------------------- 0.0/1.5 MB ? eta -:--:--\n",
      "   ----------- ---------------------------- 0.4/1.5 MB 13.4 MB/s eta 0:00:01\n",
      "   --------------- ------------------------ 0.6/1.5 MB 9.1 MB/s eta 0:00:01\n",
      "   --------------- ------------------------ 0.6/1.5 MB 9.1 MB/s eta 0:00:01\n",
      "   ------------------------- -------------- 0.9/1.5 MB 5.9 MB/s eta 0:00:01\n",
      "   --------------------------- ------------ 1.0/1.5 MB 5.4 MB/s eta 0:00:01\n",
      "   --------------------------------- ------ 1.2/1.5 MB 5.2 MB/s eta 0:00:01\n",
      "   ------------------------------------- -- 1.4/1.5 MB 4.9 MB/s eta 0:00:01\n",
      "   ------------------------------------- -- 1.4/1.5 MB 4.9 MB/s eta 0:00:01\n",
      "   ---------------------------------------- 1.5/1.5 MB 3.8 MB/s eta 0:00:00\n",
      "Downloading typer-0.12.3-py3-none-any.whl (47 kB)\n",
      "   ---------------------------------------- 0.0/47.2 kB ? eta -:--:--\n",
      "   ---------------------------------------- 47.2/47.2 kB 2.3 MB/s eta 0:00:00\n",
      "Downloading wasabi-1.1.3-py3-none-any.whl (27 kB)\n",
      "Downloading weasel-0.4.1-py3-none-any.whl (50 kB)\n",
      "   ---------------------------------------- 0.0/50.3 kB ? eta -:--:--\n",
      "   ---------------------------------------- 50.3/50.3 kB 2.5 MB/s eta 0:00:00\n",
      "Downloading blis-0.7.11-cp311-cp311-win_amd64.whl (6.6 MB)\n",
      "   ---------------------------------------- 0.0/6.6 MB ? eta -:--:--\n",
      "   ---------------------------------------- 0.0/6.6 MB 2.0 MB/s eta 0:00:04\n",
      "    --------------------------------------- 0.1/6.6 MB 1.3 MB/s eta 0:00:05\n",
      "   - -------------------------------------- 0.2/6.6 MB 1.7 MB/s eta 0:00:04\n",
      "   - -------------------------------------- 0.3/6.6 MB 1.7 MB/s eta 0:00:04\n",
      "   -- ------------------------------------- 0.5/6.6 MB 2.0 MB/s eta 0:00:04\n",
      "   --- ------------------------------------ 0.6/6.6 MB 2.1 MB/s eta 0:00:03\n",
      "   --- ------------------------------------ 0.6/6.6 MB 2.0 MB/s eta 0:00:03\n",
      "   ---- ----------------------------------- 0.7/6.6 MB 2.0 MB/s eta 0:00:03\n",
      "   ----- ---------------------------------- 0.9/6.6 MB 2.2 MB/s eta 0:00:03\n",
      "   ------- -------------------------------- 1.2/6.6 MB 2.5 MB/s eta 0:00:03\n",
      "   -------- ------------------------------- 1.4/6.6 MB 2.7 MB/s eta 0:00:02\n",
      "   --------- ------------------------------ 1.6/6.6 MB 2.9 MB/s eta 0:00:02\n",
      "   ----------- ---------------------------- 1.9/6.6 MB 3.1 MB/s eta 0:00:02\n",
      "   ------------ --------------------------- 2.1/6.6 MB 3.2 MB/s eta 0:00:02\n",
      "   -------------- ------------------------- 2.4/6.6 MB 3.3 MB/s eta 0:00:02\n",
      "   --------------- ------------------------ 2.6/6.6 MB 3.4 MB/s eta 0:00:02\n",
      "   ----------------- ---------------------- 2.8/6.6 MB 3.6 MB/s eta 0:00:02\n",
      "   ------------------ --------------------- 3.1/6.6 MB 3.7 MB/s eta 0:00:01\n",
      "   -------------------- ------------------- 3.3/6.6 MB 3.8 MB/s eta 0:00:01\n",
      "   --------------------- ------------------ 3.6/6.6 MB 3.8 MB/s eta 0:00:01\n",
      "   ----------------------- ---------------- 3.9/6.6 MB 3.9 MB/s eta 0:00:01\n",
      "   ------------------------ --------------- 4.1/6.6 MB 4.0 MB/s eta 0:00:01\n",
      "   -------------------------- ------------- 4.4/6.6 MB 4.0 MB/s eta 0:00:01\n",
      "   --------------------------- ------------ 4.6/6.6 MB 4.1 MB/s eta 0:00:01\n",
      "   ----------------------------- ---------- 4.8/6.6 MB 4.1 MB/s eta 0:00:01\n",
      "   ------------------------------- -------- 5.1/6.6 MB 4.2 MB/s eta 0:00:01\n",
      "   -------------------------------- ------- 5.4/6.6 MB 4.2 MB/s eta 0:00:01\n",
      "   ---------------------------------- ----- 5.7/6.6 MB 4.3 MB/s eta 0:00:01\n",
      "   ------------------------------------ --- 5.9/6.6 MB 4.3 MB/s eta 0:00:01\n",
      "   ------------------------------------- -- 6.2/6.6 MB 4.4 MB/s eta 0:00:01\n",
      "   ---------------------------------------  6.5/6.6 MB 4.4 MB/s eta 0:00:01\n",
      "   ---------------------------------------  6.6/6.6 MB 4.4 MB/s eta 0:00:01\n",
      "   ---------------------------------------- 6.6/6.6 MB 4.4 MB/s eta 0:00:00\n",
      "Downloading cloudpathlib-0.18.1-py3-none-any.whl (47 kB)\n",
      "   ---------------------------------------- 0.0/47.3 kB ? eta -:--:--\n",
      "   ---------------------------------------- 47.3/47.3 kB ? eta 0:00:00\n",
      "Downloading confection-0.1.5-py3-none-any.whl (35 kB)\n",
      "Downloading language_data-1.2.0-py3-none-any.whl (5.4 MB)\n",
      "   ---------------------------------------- 0.0/5.4 MB ? eta -:--:--\n",
      "   -- ------------------------------------- 0.3/5.4 MB 5.7 MB/s eta 0:00:01\n",
      "   ---- ----------------------------------- 0.6/5.4 MB 5.8 MB/s eta 0:00:01\n",
      "   ------ --------------------------------- 0.8/5.4 MB 5.8 MB/s eta 0:00:01\n",
      "   -------- ------------------------------- 1.1/5.4 MB 5.8 MB/s eta 0:00:01\n",
      "   ---------- ----------------------------- 1.4/5.4 MB 5.7 MB/s eta 0:00:01\n",
      "   ------------ --------------------------- 1.6/5.4 MB 5.8 MB/s eta 0:00:01\n",
      "   -------------- ------------------------- 1.9/5.4 MB 5.8 MB/s eta 0:00:01\n",
      "   ---------------- ----------------------- 2.2/5.4 MB 5.8 MB/s eta 0:00:01\n",
      "   ------------------ --------------------- 2.4/5.4 MB 5.8 MB/s eta 0:00:01\n",
      "   -------------------- ------------------- 2.7/5.4 MB 5.8 MB/s eta 0:00:01\n",
      "   ---------------------- ----------------- 3.0/5.4 MB 5.8 MB/s eta 0:00:01\n",
      "   ------------------------ --------------- 3.3/5.4 MB 5.8 MB/s eta 0:00:01\n",
      "   -------------------------- ------------- 3.5/5.4 MB 5.7 MB/s eta 0:00:01\n",
      "   ---------------------------- ----------- 3.8/5.4 MB 5.7 MB/s eta 0:00:01\n",
      "   ----------------------------- ---------- 4.0/5.4 MB 5.7 MB/s eta 0:00:01\n",
      "   ------------------------------- -------- 4.3/5.4 MB 5.7 MB/s eta 0:00:01\n",
      "   --------------------------------- ------ 4.6/5.4 MB 5.7 MB/s eta 0:00:01\n",
      "   ----------------------------------- ---- 4.8/5.4 MB 5.7 MB/s eta 0:00:01\n",
      "   ------------------------------------- -- 5.1/5.4 MB 5.7 MB/s eta 0:00:01\n",
      "   ---------------------------------------  5.4/5.4 MB 5.6 MB/s eta 0:00:01\n",
      "   ---------------------------------------- 5.4/5.4 MB 5.5 MB/s eta 0:00:00\n",
      "Downloading shellingham-1.5.4-py2.py3-none-any.whl (9.8 kB)\n",
      "Downloading marisa_trie-1.2.0-cp311-cp311-win_amd64.whl (152 kB)\n",
      "   ---------------------------------------- 0.0/152.6 kB ? eta -:--:--\n",
      "   ---------------------------------------- 152.6/152.6 kB 8.9 MB/s eta 0:00:00\n",
      "Installing collected packages: cymem, wasabi, spacy-loggers, spacy-legacy, shellingham, murmurhash, marisa-trie, cloudpathlib, catalogue, blis, srsly, preshed, language-data, typer, langcodes, confection, weasel, thinc, spacy\n",
      "Successfully installed blis-0.7.11 catalogue-2.0.10 cloudpathlib-0.18.1 confection-0.1.5 cymem-2.0.8 langcodes-3.4.0 language-data-1.2.0 marisa-trie-1.2.0 murmurhash-1.0.10 preshed-3.0.9 shellingham-1.5.4 spacy-3.7.5 spacy-legacy-3.0.12 spacy-loggers-1.0.5 srsly-2.4.8 thinc-8.2.5 typer-0.12.3 wasabi-1.1.3 weasel-0.4.1\n"
     ]
    }
   ],
   "source": [
    "!pip install spacy"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "6f91c9a8-c4d1-4196-9618-b15ae94953a9",
   "metadata": {},
   "outputs": [],
   "source": [
    "import re\n",
    "import nltk\n",
    "from nltk.tokenize import word_tokenize,sent_tokenize\n",
    "from nltk.corpus import indian\n",
    "from nltk.stem.porter import PorterStemmer\n",
    "from wordcloud import WordCloud\n",
    "import matplotlib.pyplot as plt\n",
    "import unicodedata\n",
    "from keras.models import Sequential,Model\n",
    "from keras.layers import Input,Embedding, LSTM, Dense\n",
    "from tensorflow.keras.preprocessing.text import Tokenizer\n",
    "from sklearn.model_selection import train_test_split\n",
    "from tensorflow.keras.preprocessing.sequence import pad_sequences\n",
    "import string\n",
    "import spacy\n",
    "import contractions\n",
    "from nltk.corpus import stopwords\n",
    "import numpy as np\n",
    "import pandas as pd "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0acde8ac-7629-42f3-ab72-ecd0b918faee",
   "metadata": {},
   "source": [
    "### Importing dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "2d130f18-75f7-43fe-b651-c129d3e47155",
   "metadata": {},
   "outputs": [],
   "source": [
    "data = pd.read_csv(\"newdata.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "b4e6708b-dc41-4d33-86d6-3e0f6c67cf06",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>english</th>\n",
       "      <th>hindi</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>On the other hand , it has been observed that ...</td>\n",
       "      <td>इसके विपरीत देखा यह गया है कि उपयुक़्त आयु में...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Now we may find that there are places that are...</td>\n",
       "      <td>अब यह भी हो सकता है कि हमें ऐसे स्थान मिलें जो...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>He has never told a lie.</td>\n",
       "      <td>उसने कभी झूठ नही बोला है.</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>I can tell you it's a magnificent spread.</td>\n",
       "      <td>मैं आप को बता सकता हूँ कि वो बहुत ही भव्य है॰</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>And one of them came to me the next morning an...</td>\n",
       "      <td>और उनमें से एक मेरे पास अगली सुबह आया और बोला,</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                            english   \\\n",
       "0  On the other hand , it has been observed that ...   \n",
       "1  Now we may find that there are places that are...   \n",
       "2                           He has never told a lie.   \n",
       "3          I can tell you it's a magnificent spread.   \n",
       "4  And one of them came to me the next morning an...   \n",
       "\n",
       "                                               hindi  \n",
       "0  इसके विपरीत देखा यह गया है कि उपयुक़्त आयु में...  \n",
       "1  अब यह भी हो सकता है कि हमें ऐसे स्थान मिलें जो...  \n",
       "2                          उसने कभी झूठ नही बोला है.  \n",
       "3      मैं आप को बता सकता हूँ कि वो बहुत ही भव्य है॰  \n",
       "4     और उनमें से एक मेरे पास अगली सुबह आया और बोला,  "
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "4697ba5b-a78f-4c80-bdee-4f0c4c864052",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 1210 entries, 0 to 1209\n",
      "Data columns (total 2 columns):\n",
      " #   Column    Non-Null Count  Dtype \n",
      "---  ------    --------------  ----- \n",
      " 0   english   1210 non-null   object\n",
      " 1   hindi     1206 non-null   object\n",
      "dtypes: object(2)\n",
      "memory usage: 19.0+ KB\n"
     ]
    }
   ],
   "source": [
    "data.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "941d59cf-9e6c-4640-b64e-f05549e68394",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>english</th>\n",
       "      <th>hindi</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>1210</td>\n",
       "      <td>1206</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>unique</th>\n",
       "      <td>1199</td>\n",
       "      <td>1194</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>top</th>\n",
       "      <td>(Laughter)</td>\n",
       "      <td>(हंसी)</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>freq</th>\n",
       "      <td>7</td>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "          english    hindi\n",
       "count         1210    1206\n",
       "unique        1199    1194\n",
       "top     (Laughter)  (हंसी)\n",
       "freq             7       4"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.describe()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1ebed989-9661-42ea-9eff-19cbfc77d990",
   "metadata": {},
   "source": [
    "### Data Preprocessing "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "ca1edf1e-fb7b-43b6-9ba9-1ade4e14077f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "english     0\n",
       "hindi       4\n",
       "dtype: int64"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "a131b078-abd4-406b-bf28-c86d5a3a8dfd",
   "metadata": {},
   "outputs": [],
   "source": [
    "data.dropna(inplace = True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "cf858f5c-ee2d-404e-9289-60ae17c43439",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "Index: 1206 entries, 0 to 1209\n",
      "Data columns (total 2 columns):\n",
      " #   Column    Non-Null Count  Dtype \n",
      "---  ------    --------------  ----- \n",
      " 0   english   1206 non-null   object\n",
      " 1   hindi     1206 non-null   object\n",
      "dtypes: object(2)\n",
      "memory usage: 28.3+ KB\n"
     ]
    }
   ],
   "source": [
    "data.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "305bb5dd-e1b2-4ced-a5bc-d8bf5eb1888d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>english</th>\n",
       "      <th>hindi</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>914</th>\n",
       "      <td>Here's the key question.</td>\n",
       "      <td>यही मुख्य सवाल है।</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>315</th>\n",
       "      <td>But I think our generation also might be the f...</td>\n",
       "      <td>मगर मैं समझता हूँ कि हमारी पीढी ने ही शायद पहल...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1070</th>\n",
       "      <td>Wisconsin</td>\n",
       "      <td>विस्कान्सिन</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>639</th>\n",
       "      <td>Unheard facts about Bhaghat Singh(Sunday)</td>\n",
       "      <td>भगत सिंह के बारे में कुछ अनदेखे तथ्य (रविवार)</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>240</th>\n",
       "      <td>after that they have do 3 year course in chose...</td>\n",
       "      <td>इसके भाद उन्हें सामान्यतया एक ३-वर्षीय स्नातक ...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>286</th>\n",
       "      <td>A couple of years ago,</td>\n",
       "      <td>दो वर्ष पहले,</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>944</th>\n",
       "      <td>Because I tried to make it intimate.</td>\n",
       "      <td>मैं उसे ये तोह्फा बड़े प्यार से देना चाहता था.</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>410</th>\n",
       "      <td>YouTube relies on its users to flag the conten...</td>\n",
       "      <td>अपलॊड करने वाला यूट्यूब कॊ अपॊड की हुइ सामग्री...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>901</th>\n",
       "      <td>At the moment though every anti-Americanist is...</td>\n",
       "      <td>वैसे , अमेरिकीपन का हर विरोधी ऐसा नहीं कर रहा ...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>405</th>\n",
       "      <td>If there is a warranty , what does it include ...</td>\n",
       "      <td>अगर वारंटी है , तो उसमें क्या शामिल है ह्यऔर क...</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                               english   \\\n",
       "914                            Here's the key question.   \n",
       "315   But I think our generation also might be the f...   \n",
       "1070                                          Wisconsin   \n",
       "639           Unheard facts about Bhaghat Singh(Sunday)   \n",
       "240   after that they have do 3 year course in chose...   \n",
       "286                              A couple of years ago,   \n",
       "944                Because I tried to make it intimate.   \n",
       "410   YouTube relies on its users to flag the conten...   \n",
       "901   At the moment though every anti-Americanist is...   \n",
       "405   If there is a warranty , what does it include ...   \n",
       "\n",
       "                                                  hindi  \n",
       "914                                  यही मुख्य सवाल है।  \n",
       "315   मगर मैं समझता हूँ कि हमारी पीढी ने ही शायद पहल...  \n",
       "1070                                        विस्कान्सिन  \n",
       "639       भगत सिंह के बारे में कुछ अनदेखे तथ्य (रविवार)  \n",
       "240   इसके भाद उन्हें सामान्यतया एक ३-वर्षीय स्नातक ...  \n",
       "286                                       दो वर्ष पहले,  \n",
       "944      मैं उसे ये तोह्फा बड़े प्यार से देना चाहता था.  \n",
       "410   अपलॊड करने वाला यूट्यूब कॊ अपॊड की हुइ सामग्री...  \n",
       "901   वैसे , अमेरिकीपन का हर विरोधी ऐसा नहीं कर रहा ...  \n",
       "405   अगर वारंटी है , तो उसमें क्या शामिल है ह्यऔर क...  "
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.sample(10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "6c37af02-f6b7-4df9-b415-0ed8a50c2adc",
   "metadata": {},
   "outputs": [],
   "source": [
    "#now we need to preprocess the text ---> remove punctuations, contractions, html tags, urls etc. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "77407441-992c-40f2-b867-6bc4d907ede6",
   "metadata": {},
   "outputs": [],
   "source": [
    "#remove html tags "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "1d332643-2d4e-4a2e-a646-273160f0a76c",
   "metadata": {},
   "outputs": [],
   "source": [
    "def remove_html(text):\n",
    "    if isinstance(text,str):\n",
    "        pattern = re.compile('<.*?>')\n",
    "        return pattern.sub(r'',text)\n",
    "    else:\n",
    "        return text"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "c4d9c4cb-6795-4485-9440-b45f825ab171",
   "metadata": {},
   "outputs": [],
   "source": [
    "# remove url"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "e5a74821-4296-407c-8117-91ca8f747197",
   "metadata": {},
   "outputs": [],
   "source": [
    "def remove_url(text):\n",
    "    if isinstance(text,str):\n",
    "        pattern = re.compile(r'https?://\\S+|www\\.\\S+')\n",
    "        return pattern.sub(r'',text)\n",
    "    else:\n",
    "        return"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "6f939bb0-0f99-4f94-aa9a-d6be2710bea6",
   "metadata": {},
   "outputs": [],
   "source": [
    "# remove alphanumeric characters"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "7dc8eae9-e639-4264-af9c-24733f430c28",
   "metadata": {},
   "outputs": [],
   "source": [
    "def preprocess_text(text, language='english'):\n",
    "    if not isinstance(text, str):\n",
    "        return text\n",
    "    if language == 'english':\n",
    "        pattern = re.compile(r'[^a-zA-Z0-9\\s]')\n",
    "        return pattern.sub(r'', text)\n",
    "    elif language == 'hindi':\n",
    "        pattern = re.compile(r'[^\\u0900-\\u097F\\s]')\n",
    "        return pattern.sub(r'', text)\n",
    "    else:\n",
    "        raise ValueError(\"Unsupported Language, Supported languages are 'english' and 'hindi'\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "9ed004cd-a417-4081-a8f9-42cf338255a3",
   "metadata": {},
   "outputs": [],
   "source": [
    "data.rename(columns = {'english ' : 'english'}, inplace = True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "d52d34f0-a112-4125-a7c9-ab0e07ed0831",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Index(['english', 'hindi'], dtype='object')"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "541042a2-8857-4c48-9b70-6ecdfce5226e",
   "metadata": {},
   "outputs": [],
   "source": [
    "data['english'] = data['english'].apply(remove_html)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "d9fcedee-1427-4a6a-8721-b7a93efca983",
   "metadata": {},
   "outputs": [],
   "source": [
    "data[\"hindi\"] = data[\"hindi\"].apply(remove_html)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "a258f6de-fb37-41db-b6f7-43045d67b1dc",
   "metadata": {},
   "outputs": [],
   "source": [
    "data['english'] = data['english'].apply(remove_url)\n",
    "data[\"hindi\"] = data[\"hindi\"].apply(remove_url)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "90257dba-4b35-46c6-9c0d-ad683ea2ed93",
   "metadata": {},
   "outputs": [],
   "source": [
    "data['english'] = data['english'].apply(lambda x: preprocess_text(x, language='english'))\n",
    "data['hindi'] = data['hindi'].apply(lambda x: preprocess_text(x, language='hindi'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "3c56bfa7-2353-4a5d-a396-d3cfd02d20cf",
   "metadata": {},
   "outputs": [],
   "source": [
    "# at this stage we don't have any html tags, url tags and alphanumeric characters"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "133903ba-c089-43d4-ab17-d9a77b38ce1b",
   "metadata": {},
   "outputs": [],
   "source": [
    "#now we need to remove punctuations before tokenising the texts "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "d453ca74-d5c9-4e9f-89ad-c42354ec036c",
   "metadata": {},
   "outputs": [],
   "source": [
    "#English punctuations"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "85a67490-27a4-408e-a3eb-1500d33e4151",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'!\"#$%&\\'()*+,-./:;<=>?@[\\\\]^_`{|}~'"
      ]
     },
     "execution_count": 25,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "string.punctuation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "92ecf253-203f-4c8e-b5bb-62a5a9569075",
   "metadata": {},
   "outputs": [],
   "source": [
    "def get_hindi_punctuations():\n",
    "    hindi_punctuations = []\n",
    "    for i in range(0x2000, 0x206f + 1):\n",
    "        char = chr(i)\n",
    "        if unicodedata.category(char) == 'Po':\n",
    "            hindi_punctuations.append(char)\n",
    "    return ''.join(hindi_punctuations)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "dd00e815-12db-44e0-984f-373de5c3652f",
   "metadata": {},
   "outputs": [],
   "source": [
    "hindi_punctuation = get_hindi_punctuations()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "2134ffb6-af6d-4c15-9b04-1ecec11e03df",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'‖‗†‡•‣․‥…‧‰‱′″‴‵‶‷‸※‼‽‾⁁⁂⁃⁇⁈⁉⁊⁋⁌⁍⁎⁏⁐⁑⁓⁕⁖⁗⁘⁙⁚⁛⁜⁝⁞'"
      ]
     },
     "execution_count": 28,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "hindi_punctuation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "10a1dc0d-3478-46a9-8bf7-deb5e0fe7611",
   "metadata": {},
   "outputs": [],
   "source": [
    "def remove_punctuation(text, language = 'english'):\n",
    "    if language == 'english':\n",
    "        exclude_english = set(string.punctuation)\n",
    "        return ''.join(char for char in text if char not in exclude_english)\n",
    "    elif language == 'hindi':\n",
    "        return ''.join(char for char in text if char not in hindi_punctuation)\n",
    "    \n",
    "    else:\n",
    "        raise ValueError(\"Unsupported Language, Supported languages are 'english' and 'hindi'\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "bf4e4c38-90ed-40a1-bd3d-a056c6e261b7",
   "metadata": {},
   "outputs": [],
   "source": [
    "data['english'] = data['english'].apply(lambda x: remove_punctuation(x,language = 'english'))\n",
    "data['hindi'] = data['hindi'].apply(lambda x: remove_punctuation(x,language = 'hindi'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "73eee57a-abfd-48fd-a2da-febe0128d376",
   "metadata": {},
   "outputs": [],
   "source": [
    "def expand_contractions(text):\n",
    "    expanded_text = contractions.fix(text)\n",
    "    return expanded_text"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "bd0d82a6-d44e-43d3-9369-83efa6151730",
   "metadata": {},
   "outputs": [],
   "source": [
    "data[\"english\"] = data[\"english\"].apply(expand_contractions)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "867cfdec-dd2b-47af-8afb-845dbbb5efda",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "id": "0c46079d-64f8-4940-99f2-ed9aa23a5a3a",
   "metadata": {},
   "outputs": [],
   "source": [
    "# data is now clean and ready to be tokenised"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "id": "8ffc69c1-4d4e-4fcb-9fa3-e509e30285f2",
   "metadata": {},
   "outputs": [],
   "source": [
    "def do_tokenization(text):\n",
    "    token_words = word_tokenize(text)\n",
    "    return token_words"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "id": "5e0f923d-6716-404b-be1d-5edb41fc6143",
   "metadata": {},
   "outputs": [],
   "source": [
    "data[\"english\"] = data[\"english\"].apply(do_tokenization)\n",
    "data[\"hindi\"] = data[\"hindi\"].apply(do_tokenization)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "id": "c7e5016f-de3b-4f69-b15a-c82f47002384",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['Hindi', 'Poet']"
      ]
     },
     "execution_count": 36,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data['english'][500]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "id": "fdde273d-4fad-45d3-a52d-17718b3ab7c5",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['हिन्दी', 'कवि']"
      ]
     },
     "execution_count": 37,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data[\"hindi\"][500]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "fa665889-4b50-47ad-af70-cd029d6c684b",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "id": "2124cf45-fc7b-4c13-844c-66352676f2c3",
   "metadata": {},
   "outputs": [],
   "source": [
    "# now amongst these we need to remove the stop words"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "id": "1b1bfb87-a224-412b-bf0a-d92aedbabfcf",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "179"
      ]
     },
     "execution_count": 39,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "len(stopwords.words(\"english\"))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "id": "54a1e816-06cf-4b22-82e0-caecd7fe669b",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[nltk_data] Downloading package indian to C:\\Users\\Diya\n",
      "[nltk_data]     Sivaprasad\\AppData\\Roaming\\nltk_data...\n",
      "[nltk_data]   Package indian is already up-to-date!\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "True"
      ]
     },
     "execution_count": 40,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "nltk.download(\"indian\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "id": "65422a9a-1cfe-4ebc-add5-175f64b98b1d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "9408"
      ]
     },
     "execution_count": 41,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "len(nltk.corpus.indian.words('hindi.pos'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "id": "0ff1ea10-b449-4a34-b117-a9c156af3cb2",
   "metadata": {},
   "outputs": [],
   "source": [
    "# collecting the stopwords"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "id": "4f8a9dc8-6e96-482d-a96e-745ab305965d",
   "metadata": {},
   "outputs": [],
   "source": [
    "stop_words_english = set(stopwords.words('english'))\n",
    "stop_words_hindi = set(nltk.corpus.indian.words('hindi.pos'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "id": "e863e775-20b1-4f26-a3a6-d3121c725623",
   "metadata": {},
   "outputs": [],
   "source": [
    "# function to remove stopwords"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "id": "1ba3c711-9209-4073-be85-cc30f7aa465c",
   "metadata": {},
   "outputs": [],
   "source": [
    "def remove_stopwords(text,language = 'english'):\n",
    "    if language == 'english':\n",
    "        filtered_words_english = [word for word in text if word.lower() not in stop_words_english]\n",
    "        return ' '.join(filtered_words_english)\n",
    "    elif language == 'hindi':\n",
    "        filterd_words_hindi = [word for word in text if word not in stop_words_hindi]\n",
    "        return ' '.join(filterd_words_hindi)\n",
    "    else:\n",
    "        return ValueError(\"Unsupported Language, Supported languages are 'english' and 'hindi'\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "id": "79600d05-c22a-42d5-9004-4fb56f194f94",
   "metadata": {},
   "outputs": [],
   "source": [
    "data['english'] = data['english'].apply(lambda x :remove_stopwords(x,language = 'english'))\n",
    "data['hindi'] = data['hindi'].apply(lambda x :remove_stopwords(x,language = 'hindi'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "id": "de43c7f7-b229-4f0d-b8dd-1ddfd55c4fdc",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>english</th>\n",
       "      <th>hindi</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>194</th>\n",
       "      <td>Data relating parliamentary activities contain...</td>\n",
       "      <td>प्रणाली डैटाबेस सम्मिलित संसदीय कार्यकलापों आध...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1208</th>\n",
       "      <td>women ages 20 64 entitled free smear test chec...</td>\n",
       "      <td>नैशनल सरवाकिल कार्यऋम महिलाएऋ ऋनकी आयु मुफ्त स...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>752</th>\n",
       "      <td>took prison governor</td>\n",
       "      <td>जेल तरह</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>803</th>\n",
       "      <td>Valmiki Ramayana Publisher Dehati Pustak Bhand...</td>\n",
       "      <td>वाल्मीकीय रामायण प्रकाशक देहाती भंडार</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>641</th>\n",
       "      <td>Traders often belong trade associations</td>\n",
       "      <td>व्यापारी अक्सर ट्रेड व्यापारी मंडलों</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                                english  \\\n",
       "194   Data relating parliamentary activities contain...   \n",
       "1208  women ages 20 64 entitled free smear test chec...   \n",
       "752                                took prison governor   \n",
       "803   Valmiki Ramayana Publisher Dehati Pustak Bhand...   \n",
       "641             Traders often belong trade associations   \n",
       "\n",
       "                                                  hindi  \n",
       "194   प्रणाली डैटाबेस सम्मिलित संसदीय कार्यकलापों आध...  \n",
       "1208  नैशनल सरवाकिल कार्यऋम महिलाएऋ ऋनकी आयु मुफ्त स...  \n",
       "752                                             जेल तरह  \n",
       "803               वाल्मीकीय रामायण प्रकाशक देहाती भंडार  \n",
       "641                व्यापारी अक्सर ट्रेड व्यापारी मंडलों  "
      ]
     },
     "execution_count": 47,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.sample(5)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "id": "917613e4-104f-47a7-b277-40403490b19a",
   "metadata": {},
   "outputs": [],
   "source": [
    "# now we need to convert words into their root forms ---> via the process of stemming"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "id": "3d8fff7a-84a4-47d0-9f37-6f65a4d67e6b",
   "metadata": {},
   "outputs": [],
   "source": [
    "def stemming(text):\n",
    "    ps = PorterStemmer()\n",
    "    words = text.split()\n",
    "    return [ps.stem(word) for word in words]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "id": "42a1b22f-c8f7-435a-8387-9c5a0dc2257c",
   "metadata": {},
   "outputs": [],
   "source": [
    "data[\"english\"] = data[\"english\"].apply(stemming)\n",
    "data[\"hindi\"] = data[\"hindi\"].apply(stemming)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "id": "cc62b8b7-4438-4f43-8909-7d1345a69b83",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>english</th>\n",
       "      <th>hindi</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>831</th>\n",
       "      <td>[relat, part]</td>\n",
       "      <td>[कड़ियाँ]</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>110</th>\n",
       "      <td>[context]</td>\n",
       "      <td>[संदर्भ]</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>728</th>\n",
       "      <td>[rememb, benefit, gp, good, night, sleep, call...</td>\n",
       "      <td>[याद, रखें, आपको, जभी, लाभ, आपके, जीपी, रातभर,...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>37</th>\n",
       "      <td>[typic, second, best, guess, third, best, gues...</td>\n",
       "      <td>[आमतौर, पूर्वानुमान, जवाब, था।]</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>408</th>\n",
       "      <td>[feel, love, bind, us]</td>\n",
       "      <td>[एकदूसरे, जोड़नेवाला, प्रेम]</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                               english  \\\n",
       "831                                      [relat, part]   \n",
       "110                                          [context]   \n",
       "728  [rememb, benefit, gp, good, night, sleep, call...   \n",
       "37   [typic, second, best, guess, third, best, gues...   \n",
       "408                             [feel, love, bind, us]   \n",
       "\n",
       "                                                 hindi  \n",
       "831                                          [कड़ियाँ]  \n",
       "110                                           [संदर्भ]  \n",
       "728  [याद, रखें, आपको, जभी, लाभ, आपके, जीपी, रातभर,...  \n",
       "37                     [आमतौर, पूर्वानुमान, जवाब, था।]  \n",
       "408                       [एकदूसरे, जोड़नेवाला, प्रेम]  "
      ]
     },
     "execution_count": 51,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.sample(5)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "id": "a292bab7-7773-4a87-859e-3fdc77330d84",
   "metadata": {},
   "outputs": [],
   "source": [
    "# we are done with the data pre-processing "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "bde9cb29-13f5-4932-a863-18005d55052f",
   "metadata": {},
   "source": [
    "### Tokenisation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "id": "daa3c552-cd62-4bb9-b393-f8e638191f9f",
   "metadata": {},
   "outputs": [],
   "source": [
    "tokenizer_english = Tokenizer()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "id": "cc9e6cd9-7fd1-48bd-8261-fba7e31b38c4",
   "metadata": {},
   "outputs": [],
   "source": [
    "tokenizer_english.fit_on_texts(data[\"english\"])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "id": "bd8d7e29-831a-431b-b657-5306bc901e5e",
   "metadata": {},
   "outputs": [],
   "source": [
    "tokenizer_hindi = Tokenizer()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "id": "400a9287-de17-4282-b732-d29338a6e732",
   "metadata": {},
   "outputs": [],
   "source": [
    "tokenizer_hindi.fit_on_texts(data[\"hindi\"])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "id": "a7393930-9621-4dae-8a64-9af087decb79",
   "metadata": {},
   "outputs": [],
   "source": [
    "tokenizer_hindi.word_index['<start>'] = len(tokenizer_hindi.word_index) + 1\n",
    "tokenizer_hindi.word_index['<end>'] = len(tokenizer_hindi.word_index) + 1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "id": "7c6506ea-50cb-46bb-887b-4f5594ea1eff",
   "metadata": {},
   "outputs": [],
   "source": [
    "def add_special_tokens(sequences, start_token='<start>', end_token='<end>'):\n",
    "    sequences_with_special_tokens = []\n",
    "    for sequence in sequences:\n",
    "        sequence_with_special_tokens = [start_token] + sequence + [end_token]\n",
    "        sequences_with_special_tokens.append(sequence_with_special_tokens)\n",
    "    return sequences_with_special_tokens"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "id": "903e8aea-42cd-467c-a6b0-268c4ee60df0",
   "metadata": {},
   "outputs": [],
   "source": [
    "data['hindi'] = add_special_tokens(data['hindi'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "id": "5ebc20d0-3d6f-4494-a2b4-943c52032487",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>english</th>\n",
       "      <th>hindi</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>1040</th>\n",
       "      <td>[whole, detail, go, happen, polit, scene]</td>\n",
       "      <td>[&lt;start&gt;, यंहा, &lt;end&gt;]</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>772</th>\n",
       "      <td>[left, unknown, destin, abroad]</td>\n",
       "      <td>[&lt;start&gt;, अज्ञात, चुपचाप, प्रस्थान, करा, &lt;end&gt;]</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1161</th>\n",
       "      <td>[south, korea, 185]</td>\n",
       "      <td>[&lt;start&gt;, कोरिया, १८५, &lt;end&gt;]</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>442</th>\n",
       "      <td>[upset]</td>\n",
       "      <td>[&lt;start&gt;, तुम, जाओगी, &lt;end&gt;]</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>717</th>\n",
       "      <td>[music]</td>\n",
       "      <td>[&lt;start&gt;, &lt;end&gt;]</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>481</th>\n",
       "      <td>[directli, approach, princip, scottish, church...</td>\n",
       "      <td>[&lt;start&gt;, सीधे, स्काटिश, चर्च, कालेज, प्रिंसिप...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>412</th>\n",
       "      <td>[meanwhil, 15, januari, north, india, rock, ea...</td>\n",
       "      <td>[&lt;start&gt;, उत्तरी, जनपदों, संपत्ति, क्षति, &lt;end&gt;]</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>827</th>\n",
       "      <td>[accord, historian, viewpointdur, end, civilis...</td>\n",
       "      <td>[&lt;start&gt;, इतिहासकारों, दृष्टिकोण, सभ्यता, अन्त...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>319</th>\n",
       "      <td>[civil, societi, leader]</td>\n",
       "      <td>[&lt;start&gt;, सामजिक, लगायें, &lt;end&gt;]</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1005</th>\n",
       "      <td>[laughter]</td>\n",
       "      <td>[&lt;start&gt;, ठहाके, &lt;end&gt;]</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                                english  \\\n",
       "1040          [whole, detail, go, happen, polit, scene]   \n",
       "772                     [left, unknown, destin, abroad]   \n",
       "1161                                [south, korea, 185]   \n",
       "442                                             [upset]   \n",
       "717                                             [music]   \n",
       "481   [directli, approach, princip, scottish, church...   \n",
       "412   [meanwhil, 15, januari, north, india, rock, ea...   \n",
       "827   [accord, historian, viewpointdur, end, civilis...   \n",
       "319                            [civil, societi, leader]   \n",
       "1005                                         [laughter]   \n",
       "\n",
       "                                                  hindi  \n",
       "1040                             [<start>, यंहा, <end>]  \n",
       "772     [<start>, अज्ञात, चुपचाप, प्रस्थान, करा, <end>]  \n",
       "1161                      [<start>, कोरिया, १८५, <end>]  \n",
       "442                        [<start>, तुम, जाओगी, <end>]  \n",
       "717                                    [<start>, <end>]  \n",
       "481   [<start>, सीधे, स्काटिश, चर्च, कालेज, प्रिंसिप...  \n",
       "412    [<start>, उत्तरी, जनपदों, संपत्ति, क्षति, <end>]  \n",
       "827   [<start>, इतिहासकारों, दृष्टिकोण, सभ्यता, अन्त...  \n",
       "319                    [<start>, सामजिक, लगायें, <end>]  \n",
       "1005                            [<start>, ठहाके, <end>]  "
      ]
     },
     "execution_count": 60,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.sample(10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "id": "a01a96fc-c228-43f8-8388-d0da5338f391",
   "metadata": {},
   "outputs": [],
   "source": [
    "data[\"english\"] = tokenizer_english.texts_to_sequences(data[\"english\"])\n",
    "data[\"hindi\"] = tokenizer_hindi.texts_to_sequences(data[\"hindi\"])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 62,
   "id": "267fc666-3d02-4214-8ab9-5d1c73873524",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>english</th>\n",
       "      <th>hindi</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>582</th>\n",
       "      <td>[2854, 301]</td>\n",
       "      <td>[4582, 2929, 2930, 2931, 4583]</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>917</th>\n",
       "      <td>[195, 9, 3476, 3477]</td>\n",
       "      <td>[4582, 82, 3794, 3795, 253, 4583]</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>317</th>\n",
       "      <td>[844, 2296, 4, 1251, 5, 468, 2297, 2298, 12, 8...</td>\n",
       "      <td>[4582, 2135, 2136, 830, 730, 2137, 151, 2138, ...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>925</th>\n",
       "      <td>[36, 322, 38, 381, 804, 491, 10]</td>\n",
       "      <td>[4582, 506, 5, 3819, 161, 3820, 4583]</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>155</th>\n",
       "      <td>[769, 387, 86, 272, 96, 571, 1106, 572, 1911]</td>\n",
       "      <td>[4582, 82, 386, 1610, 399, 678, 4583]</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1060</th>\n",
       "      <td>[969, 47, 3722, 104, 47, 3723, 68, 336]</td>\n",
       "      <td>[4582, 1029, 1132, 1132, 4167, 4583]</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>707</th>\n",
       "      <td>[219, 93, 63, 142]</td>\n",
       "      <td>[4582, 392, 40, 1027, 4583]</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>656</th>\n",
       "      <td>[1164, 2987, 2988, 2989, 454, 2990, 2991, 898,...</td>\n",
       "      <td>[4582, 958, 3123, 3124, 3125, 1012, 3126, 4583]</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1198</th>\n",
       "      <td>[4002, 70, 591, 537, 626, 402, 296, 448]</td>\n",
       "      <td>[4582, 4525, 4526, 4527, 4528, 235, 105, 4583]</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>983</th>\n",
       "      <td>[326, 524, 1462, 216, 211, 1444, 1555, 3589, 3...</td>\n",
       "      <td>[4582, 607, 3960, 3961, 3962, 398, 4583]</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                                english  \\\n",
       "582                                         [2854, 301]   \n",
       "917                                [195, 9, 3476, 3477]   \n",
       "317   [844, 2296, 4, 1251, 5, 468, 2297, 2298, 12, 8...   \n",
       "925                    [36, 322, 38, 381, 804, 491, 10]   \n",
       "155       [769, 387, 86, 272, 96, 571, 1106, 572, 1911]   \n",
       "1060            [969, 47, 3722, 104, 47, 3723, 68, 336]   \n",
       "707                                  [219, 93, 63, 142]   \n",
       "656   [1164, 2987, 2988, 2989, 454, 2990, 2991, 898,...   \n",
       "1198           [4002, 70, 591, 537, 626, 402, 296, 448]   \n",
       "983   [326, 524, 1462, 216, 211, 1444, 1555, 3589, 3...   \n",
       "\n",
       "                                                  hindi  \n",
       "582                      [4582, 2929, 2930, 2931, 4583]  \n",
       "917                   [4582, 82, 3794, 3795, 253, 4583]  \n",
       "317   [4582, 2135, 2136, 830, 730, 2137, 151, 2138, ...  \n",
       "925               [4582, 506, 5, 3819, 161, 3820, 4583]  \n",
       "155               [4582, 82, 386, 1610, 399, 678, 4583]  \n",
       "1060               [4582, 1029, 1132, 1132, 4167, 4583]  \n",
       "707                         [4582, 392, 40, 1027, 4583]  \n",
       "656     [4582, 958, 3123, 3124, 3125, 1012, 3126, 4583]  \n",
       "1198     [4582, 4525, 4526, 4527, 4528, 235, 105, 4583]  \n",
       "983            [4582, 607, 3960, 3961, 3962, 398, 4583]  "
      ]
     },
     "execution_count": 62,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.sample(10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "id": "1888502a-b9df-41f4-bcd5-9a581f34ccb3",
   "metadata": {},
   "outputs": [],
   "source": [
    "#now we can go ahead with the train and test split part"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9a51ffd9-f630-4d0e-b6c3-fa46228a65f2",
   "metadata": {},
   "source": [
    "### Model Training "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 64,
   "id": "dc5bcc64-6a73-4bcc-8155-73e3d2e2a1fb",
   "metadata": {},
   "outputs": [],
   "source": [
    "x_train, x_test, y_train, y_test = train_test_split(data[\"english\"], data[\"hindi\"], test_size = 0.2, random_state = 42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 65,
   "id": "9664ba00-2020-4b74-82e9-d7b33f0323fb",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "964\n",
      "242\n"
     ]
    }
   ],
   "source": [
    "print(len(x_train))\n",
    "print(len(x_test))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 66,
   "id": "65d50182-3204-479e-8892-2664e2cea4f1",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "964\n",
      "242\n"
     ]
    }
   ],
   "source": [
    "print(len(y_train))\n",
    "print(len(y_test))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 67,
   "id": "c43206cf-411f-4df9-95ee-c73953364a95",
   "metadata": {},
   "outputs": [],
   "source": [
    "# now we add padding to the data "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 68,
   "id": "83771363-dc66-4cd2-92e4-909a06b5f744",
   "metadata": {},
   "outputs": [],
   "source": [
    "def max_sequence_length(text):\n",
    "    combined_sequences = text\n",
    "    max_length_combined = max(len(sequence) for sequence in combined_sequences)\n",
    "    return max_length_combined"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 69,
   "id": "61ef6065-f194-4fe7-9e1c-7a2bde9b3a6c",
   "metadata": {},
   "outputs": [],
   "source": [
    "max_length_x = max_sequence_length(data[\"english\"])\n",
    "max_length_y = max_sequence_length(data[\"hindi\"])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 70,
   "id": "f0d71e9d-7902-4d08-bd1a-377d4ee1cc5e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "124"
      ]
     },
     "execution_count": 70,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "max_length_x"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 71,
   "id": "a127919a-99cd-4b8c-9fce-b07569d6a33c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "92"
      ]
     },
     "execution_count": 71,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "max_length_y"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 72,
   "id": "0e1d180d-0425-4823-ba97-7e7f19d3e0cd",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(964,)"
      ]
     },
     "execution_count": 72,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x_train.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 73,
   "id": "93cc61d7-2b3b-4272-892b-6ddfb4eb2440",
   "metadata": {},
   "outputs": [],
   "source": [
    "x_train_new = pad_sequences(x_train, maxlen = max_length_x, padding = 'post')\n",
    "y_train_new = pad_sequences(y_train, maxlen = max_length_y, padding = 'post')\n",
    "x_test_new = pad_sequences(x_test, maxlen = max_length_x, padding = 'post')\n",
    "y_test_new = pad_sequences(y_test, maxlen = max_length_y, padding = 'post')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d94e07a3-7ae6-4451-acf3-bb048cc8a474",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 74,
   "id": "33dd5d5e-d627-4350-a2f2-d5b3c0ac4b8e",
   "metadata": {},
   "outputs": [],
   "source": [
    "#establishing the model "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 89,
   "id": "cad6415d-2a57-4738-95f4-39953b7dde3c",
   "metadata": {},
   "outputs": [],
   "source": [
    "import tensorflow as tf\n",
    "from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Attention,Activation, Concatenate, TimeDistributed, Dot"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 76,
   "id": "9208a9d4-bcff-4c4f-a629-f796dab4f6d4",
   "metadata": {},
   "outputs": [],
   "source": [
    "from tensorflow.keras.models import Model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 91,
   "id": "4a570aff-9afe-4c3f-8437-23b83d59791c",
   "metadata": {},
   "outputs": [],
   "source": [
    "def create_model(vocab_size_input, vocab_size_output, max_seq_length_input, max_seq_length_output, embedding_dim, hidden_units):\n",
    "    ## encoder\n",
    "    encoder_inputs = Input(shape=(max_seq_length_input,))\n",
    "    encoder_embedding = Embedding(input_dim=vocab_size_input, output_dim=embedding_dim)(encoder_inputs)\n",
    "    encoder_lstm, state_h, state_c = LSTM(hidden_units, return_state=True, return_sequences = True)(encoder_embedding)\n",
    "    encoder_states = [state_h, state_c]\n",
    "    \n",
    "    ## decoder\n",
    "    decoder_inputs = Input(shape=(max_seq_length_output,))\n",
    "    decoder_embedding = Embedding(input_dim=vocab_size_output, output_dim=embedding_dim)(decoder_inputs)\n",
    "    decoder_lstm = LSTM(hidden_units, return_sequences=True, return_state=True)\n",
    "    decoder_lstm_output, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)\n",
    "\n",
    "    ##attention\n",
    "    attention_dot = Dot(axes=[2, 2])([decoder_lstm_output, encoder_lstm])\n",
    "    attention_activation = Activation('softmax')(attention_dot)\n",
    "    context_vector = Dot(axes=[2, 1])([attention_activation, encoder_lstm])\n",
    "    attention_output = Concatenate(axis=-1)([context_vector, decoder_lstm_output])\n",
    "\n",
    "    ##output\n",
    "    decoder_dense = TimeDistributed(Dense(vocab_size_output, activation='softmax'))\n",
    "    decoder_outputs = decoder_dense(attention_output)\n",
    "    \n",
    "    ## model\n",
    "    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)\n",
    "    print(model.summary())\n",
    "    return model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 78,
   "id": "fb2876de-6ef6-40a6-9df6-6fd01234ce26",
   "metadata": {},
   "outputs": [],
   "source": [
    "vocab_size_input = len(tokenizer_english.word_index) + 1\n",
    "vocab_size_output = len(tokenizer_hindi.word_index) + 1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 79,
   "id": "0588a177-0f49-4e50-b55b-95029e22791e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "4584"
      ]
     },
     "execution_count": 79,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "vocab_size_output"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 80,
   "id": "17022bbd-ccdc-4ba6-be4b-ced614a3cef7",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "4044"
      ]
     },
     "execution_count": 80,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "vocab_size_input"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 81,
   "id": "7783bbb0-ec04-4324-968d-c6a3c7411c1b",
   "metadata": {},
   "outputs": [],
   "source": [
    "max_seq_length_input = max_length_x\n",
    "max_seq_length_output = max_length_y - 1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 82,
   "id": "285b169a-1dcc-4120-8716-fbeb43f826da",
   "metadata": {},
   "outputs": [],
   "source": [
    "# defining other hyper-parameters"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 83,
   "id": "42ed384e-3bd7-4748-a75e-7fb9793267a4",
   "metadata": {},
   "outputs": [],
   "source": [
    "embedding_dim = 100\n",
    "hidden_units = 256"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 92,
   "id": "4c1332f2-191f-4aa6-85a6-ee6704985e5e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\"><span style=\"font-weight: bold\">Model: \"functional_1\"</span>\n",
       "</pre>\n"
      ],
      "text/plain": [
       "\u001b[1mModel: \"functional_1\"\u001b[0m\n"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\">┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┓\n",
       "┃<span style=\"font-weight: bold\"> Layer (type)        </span>┃<span style=\"font-weight: bold\"> Output Shape      </span>┃<span style=\"font-weight: bold\">    Param # </span>┃<span style=\"font-weight: bold\"> Connected to      </span>┃\n",
       "┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━┩\n",
       "│ input_layer_8       │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">124</span>)       │          <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> │ -                 │\n",
       "│ (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">InputLayer</span>)        │                   │            │                   │\n",
       "├─────────────────────┼───────────────────┼────────────┼───────────────────┤\n",
       "│ input_layer_9       │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">91</span>)        │          <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> │ -                 │\n",
       "│ (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">InputLayer</span>)        │                   │            │                   │\n",
       "├─────────────────────┼───────────────────┼────────────┼───────────────────┤\n",
       "│ embedding_8         │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">124</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">100</span>)  │    <span style=\"color: #00af00; text-decoration-color: #00af00\">404,400</span> │ input_layer_8[<span style=\"color: #00af00; text-decoration-color: #00af00\">0</span>]… │\n",
       "│ (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Embedding</span>)         │                   │            │                   │\n",
       "├─────────────────────┼───────────────────┼────────────┼───────────────────┤\n",
       "│ embedding_9         │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">91</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">100</span>)   │    <span style=\"color: #00af00; text-decoration-color: #00af00\">458,400</span> │ input_layer_9[<span style=\"color: #00af00; text-decoration-color: #00af00\">0</span>]… │\n",
       "│ (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Embedding</span>)         │                   │            │                   │\n",
       "├─────────────────────┼───────────────────┼────────────┼───────────────────┤\n",
       "│ lstm_8 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">LSTM</span>)       │ [(<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">124</span>,      │    <span style=\"color: #00af00; text-decoration-color: #00af00\">365,568</span> │ embedding_8[<span style=\"color: #00af00; text-decoration-color: #00af00\">0</span>][<span style=\"color: #00af00; text-decoration-color: #00af00\">0</span>] │\n",
       "│                     │ <span style=\"color: #00af00; text-decoration-color: #00af00\">256</span>), (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>,      │            │                   │\n",
       "│                     │ <span style=\"color: #00af00; text-decoration-color: #00af00\">256</span>), (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>,      │            │                   │\n",
       "│                     │ <span style=\"color: #00af00; text-decoration-color: #00af00\">256</span>)]             │            │                   │\n",
       "├─────────────────────┼───────────────────┼────────────┼───────────────────┤\n",
       "│ lstm_9 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">LSTM</span>)       │ [(<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">91</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">256</span>), │    <span style=\"color: #00af00; text-decoration-color: #00af00\">365,568</span> │ embedding_9[<span style=\"color: #00af00; text-decoration-color: #00af00\">0</span>][<span style=\"color: #00af00; text-decoration-color: #00af00\">0</span>… │\n",
       "│                     │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">256</span>),      │            │ lstm_8[<span style=\"color: #00af00; text-decoration-color: #00af00\">0</span>][<span style=\"color: #00af00; text-decoration-color: #00af00\">1</span>],     │\n",
       "│                     │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">256</span>)]      │            │ lstm_8[<span style=\"color: #00af00; text-decoration-color: #00af00\">0</span>][<span style=\"color: #00af00; text-decoration-color: #00af00\">2</span>]      │\n",
       "├─────────────────────┼───────────────────┼────────────┼───────────────────┤\n",
       "│ dot_3 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Dot</span>)         │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">91</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">124</span>)   │          <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> │ lstm_9[<span style=\"color: #00af00; text-decoration-color: #00af00\">0</span>][<span style=\"color: #00af00; text-decoration-color: #00af00\">0</span>],     │\n",
       "│                     │                   │            │ lstm_8[<span style=\"color: #00af00; text-decoration-color: #00af00\">0</span>][<span style=\"color: #00af00; text-decoration-color: #00af00\">0</span>]      │\n",
       "├─────────────────────┼───────────────────┼────────────┼───────────────────┤\n",
       "│ activation_1        │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">91</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">124</span>)   │          <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> │ dot_3[<span style=\"color: #00af00; text-decoration-color: #00af00\">0</span>][<span style=\"color: #00af00; text-decoration-color: #00af00\">0</span>]       │\n",
       "│ (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Activation</span>)        │                   │            │                   │\n",
       "├─────────────────────┼───────────────────┼────────────┼───────────────────┤\n",
       "│ dot_4 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Dot</span>)         │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">91</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">256</span>)   │          <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> │ activation_1[<span style=\"color: #00af00; text-decoration-color: #00af00\">0</span>][<span style=\"color: #00af00; text-decoration-color: #00af00\">…</span> │\n",
       "│                     │                   │            │ lstm_8[<span style=\"color: #00af00; text-decoration-color: #00af00\">0</span>][<span style=\"color: #00af00; text-decoration-color: #00af00\">0</span>]      │\n",
       "├─────────────────────┼───────────────────┼────────────┼───────────────────┤\n",
       "│ concatenate_1       │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">91</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">512</span>)   │          <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> │ dot_4[<span style=\"color: #00af00; text-decoration-color: #00af00\">0</span>][<span style=\"color: #00af00; text-decoration-color: #00af00\">0</span>],      │\n",
       "│ (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Concatenate</span>)       │                   │            │ lstm_9[<span style=\"color: #00af00; text-decoration-color: #00af00\">0</span>][<span style=\"color: #00af00; text-decoration-color: #00af00\">0</span>]      │\n",
       "├─────────────────────┼───────────────────┼────────────┼───────────────────┤\n",
       "│ time_distributed_1  │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">91</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">4584</span>)  │  <span style=\"color: #00af00; text-decoration-color: #00af00\">2,351,592</span> │ concatenate_1[<span style=\"color: #00af00; text-decoration-color: #00af00\">0</span>]… │\n",
       "│ (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">TimeDistributed</span>)   │                   │            │                   │\n",
       "└─────────────────────┴───────────────────┴────────────┴───────────────────┘\n",
       "</pre>\n"
      ],
      "text/plain": [
       "┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┓\n",
       "┃\u001b[1m \u001b[0m\u001b[1mLayer (type)       \u001b[0m\u001b[1m \u001b[0m┃\u001b[1m \u001b[0m\u001b[1mOutput Shape     \u001b[0m\u001b[1m \u001b[0m┃\u001b[1m \u001b[0m\u001b[1m   Param #\u001b[0m\u001b[1m \u001b[0m┃\u001b[1m \u001b[0m\u001b[1mConnected to     \u001b[0m\u001b[1m \u001b[0m┃\n",
       "┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━┩\n",
       "│ input_layer_8       │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m124\u001b[0m)       │          \u001b[38;5;34m0\u001b[0m │ -                 │\n",
       "│ (\u001b[38;5;33mInputLayer\u001b[0m)        │                   │            │                   │\n",
       "├─────────────────────┼───────────────────┼────────────┼───────────────────┤\n",
       "│ input_layer_9       │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m91\u001b[0m)        │          \u001b[38;5;34m0\u001b[0m │ -                 │\n",
       "│ (\u001b[38;5;33mInputLayer\u001b[0m)        │                   │            │                   │\n",
       "├─────────────────────┼───────────────────┼────────────┼───────────────────┤\n",
       "│ embedding_8         │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m124\u001b[0m, \u001b[38;5;34m100\u001b[0m)  │    \u001b[38;5;34m404,400\u001b[0m │ input_layer_8[\u001b[38;5;34m0\u001b[0m]… │\n",
       "│ (\u001b[38;5;33mEmbedding\u001b[0m)         │                   │            │                   │\n",
       "├─────────────────────┼───────────────────┼────────────┼───────────────────┤\n",
       "│ embedding_9         │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m91\u001b[0m, \u001b[38;5;34m100\u001b[0m)   │    \u001b[38;5;34m458,400\u001b[0m │ input_layer_9[\u001b[38;5;34m0\u001b[0m]… │\n",
       "│ (\u001b[38;5;33mEmbedding\u001b[0m)         │                   │            │                   │\n",
       "├─────────────────────┼───────────────────┼────────────┼───────────────────┤\n",
       "│ lstm_8 (\u001b[38;5;33mLSTM\u001b[0m)       │ [(\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m124\u001b[0m,      │    \u001b[38;5;34m365,568\u001b[0m │ embedding_8[\u001b[38;5;34m0\u001b[0m][\u001b[38;5;34m0\u001b[0m] │\n",
       "│                     │ \u001b[38;5;34m256\u001b[0m), (\u001b[38;5;45mNone\u001b[0m,      │            │                   │\n",
       "│                     │ \u001b[38;5;34m256\u001b[0m), (\u001b[38;5;45mNone\u001b[0m,      │            │                   │\n",
       "│                     │ \u001b[38;5;34m256\u001b[0m)]             │            │                   │\n",
       "├─────────────────────┼───────────────────┼────────────┼───────────────────┤\n",
       "│ lstm_9 (\u001b[38;5;33mLSTM\u001b[0m)       │ [(\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m91\u001b[0m, \u001b[38;5;34m256\u001b[0m), │    \u001b[38;5;34m365,568\u001b[0m │ embedding_9[\u001b[38;5;34m0\u001b[0m][\u001b[38;5;34m0\u001b[0m… │\n",
       "│                     │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m256\u001b[0m),      │            │ lstm_8[\u001b[38;5;34m0\u001b[0m][\u001b[38;5;34m1\u001b[0m],     │\n",
       "│                     │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m256\u001b[0m)]      │            │ lstm_8[\u001b[38;5;34m0\u001b[0m][\u001b[38;5;34m2\u001b[0m]      │\n",
       "├─────────────────────┼───────────────────┼────────────┼───────────────────┤\n",
       "│ dot_3 (\u001b[38;5;33mDot\u001b[0m)         │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m91\u001b[0m, \u001b[38;5;34m124\u001b[0m)   │          \u001b[38;5;34m0\u001b[0m │ lstm_9[\u001b[38;5;34m0\u001b[0m][\u001b[38;5;34m0\u001b[0m],     │\n",
       "│                     │                   │            │ lstm_8[\u001b[38;5;34m0\u001b[0m][\u001b[38;5;34m0\u001b[0m]      │\n",
       "├─────────────────────┼───────────────────┼────────────┼───────────────────┤\n",
       "│ activation_1        │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m91\u001b[0m, \u001b[38;5;34m124\u001b[0m)   │          \u001b[38;5;34m0\u001b[0m │ dot_3[\u001b[38;5;34m0\u001b[0m][\u001b[38;5;34m0\u001b[0m]       │\n",
       "│ (\u001b[38;5;33mActivation\u001b[0m)        │                   │            │                   │\n",
       "├─────────────────────┼───────────────────┼────────────┼───────────────────┤\n",
       "│ dot_4 (\u001b[38;5;33mDot\u001b[0m)         │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m91\u001b[0m, \u001b[38;5;34m256\u001b[0m)   │          \u001b[38;5;34m0\u001b[0m │ activation_1[\u001b[38;5;34m0\u001b[0m][\u001b[38;5;34m…\u001b[0m │\n",
       "│                     │                   │            │ lstm_8[\u001b[38;5;34m0\u001b[0m][\u001b[38;5;34m0\u001b[0m]      │\n",
       "├─────────────────────┼───────────────────┼────────────┼───────────────────┤\n",
       "│ concatenate_1       │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m91\u001b[0m, \u001b[38;5;34m512\u001b[0m)   │          \u001b[38;5;34m0\u001b[0m │ dot_4[\u001b[38;5;34m0\u001b[0m][\u001b[38;5;34m0\u001b[0m],      │\n",
       "│ (\u001b[38;5;33mConcatenate\u001b[0m)       │                   │            │ lstm_9[\u001b[38;5;34m0\u001b[0m][\u001b[38;5;34m0\u001b[0m]      │\n",
       "├─────────────────────┼───────────────────┼────────────┼───────────────────┤\n",
       "│ time_distributed_1  │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m91\u001b[0m, \u001b[38;5;34m4584\u001b[0m)  │  \u001b[38;5;34m2,351,592\u001b[0m │ concatenate_1[\u001b[38;5;34m0\u001b[0m]… │\n",
       "│ (\u001b[38;5;33mTimeDistributed\u001b[0m)   │                   │            │                   │\n",
       "└─────────────────────┴───────────────────┴────────────┴───────────────────┘\n"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\"><span style=\"font-weight: bold\"> Total params: </span><span style=\"color: #00af00; text-decoration-color: #00af00\">3,945,528</span> (15.05 MB)\n",
       "</pre>\n"
      ],
      "text/plain": [
       "\u001b[1m Total params: \u001b[0m\u001b[38;5;34m3,945,528\u001b[0m (15.05 MB)\n"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\"><span style=\"font-weight: bold\"> Trainable params: </span><span style=\"color: #00af00; text-decoration-color: #00af00\">3,945,528</span> (15.05 MB)\n",
       "</pre>\n"
      ],
      "text/plain": [
       "\u001b[1m Trainable params: \u001b[0m\u001b[38;5;34m3,945,528\u001b[0m (15.05 MB)\n"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\"><span style=\"font-weight: bold\"> Non-trainable params: </span><span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> (0.00 B)\n",
       "</pre>\n"
      ],
      "text/plain": [
       "\u001b[1m Non-trainable params: \u001b[0m\u001b[38;5;34m0\u001b[0m (0.00 B)\n"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "None\n"
     ]
    }
   ],
   "source": [
    "model = create_model(vocab_size_input, vocab_size_output, max_seq_length_input, max_seq_length_output, embedding_dim, hidden_units)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 93,
   "id": "d8a644b6-bdbe-4fd9-a8f0-f1484215491b",
   "metadata": {},
   "outputs": [],
   "source": [
    "#to note that the embedding and the dense layers contribute to the maximum number of parameters"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 94,
   "id": "b6432a7d-7680-47d8-b3a2-298748c63ec8",
   "metadata": {},
   "outputs": [],
   "source": [
    "#we have the model now to compile and fit on the data "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 95,
   "id": "abb8a105-2ad4-4051-823d-6d4a36bbae85",
   "metadata": {},
   "outputs": [],
   "source": [
    "model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 96,
   "id": "34c8da26-b506-4df5-92ba-966a87af75bb",
   "metadata": {},
   "outputs": [],
   "source": [
    "#define callbacks and checkpoints"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 97,
   "id": "860bf125-d47b-4e23-b4a2-00842c3b7b0d",
   "metadata": {},
   "outputs": [],
   "source": [
    "from keras.callbacks import EarlyStopping, ModelCheckpoint"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 98,
   "id": "fc8d5686-c283-4bdf-89c0-768321391d18",
   "metadata": {},
   "outputs": [],
   "source": [
    "callbacks = [\n",
    "    EarlyStopping(patience = 3, monitor = \"val_loss\"),\n",
    "    ModelCheckpoint(filepath='model_weights.weights.h5', save_best_only=True, save_weights_only=True, monitor='val_loss')\n",
    "]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 99,
   "id": "96d17474-8524-4f32-a233-2586902eba9a",
   "metadata": {},
   "outputs": [],
   "source": [
    "batch_size = 60\n",
    "epochs = 20"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 100,
   "id": "cbc8b499-53fc-4bae-b7e9-febb141faadb",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "964"
      ]
     },
     "execution_count": 100,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "len(y_train_new)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 101,
   "id": "e51c8f9f-a596-4de2-8d97-ab688a628c40",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/10\n",
      "\u001b[1m17/17\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m44s\u001b[0m 2s/step - accuracy: 0.7421 - loss: 5.6442 - val_accuracy: 0.9231 - val_loss: 0.7170\n",
      "Epoch 2/10\n",
      "\u001b[1m17/17\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m25s\u001b[0m 1s/step - accuracy: 0.9202 - loss: 0.7183 - val_accuracy: 0.9231 - val_loss: 0.6756\n",
      "Epoch 3/10\n",
      "\u001b[1m17/17\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m27s\u001b[0m 2s/step - accuracy: 0.9208 - loss: 0.6343 - val_accuracy: 0.9232 - val_loss: 0.6709\n",
      "Epoch 4/10\n",
      "\u001b[1m17/17\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m27s\u001b[0m 2s/step - accuracy: 0.9194 - loss: 0.6274 - val_accuracy: 0.9268 - val_loss: 0.6766\n",
      "Epoch 5/10\n",
      "\u001b[1m17/17\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m26s\u001b[0m 2s/step - accuracy: 0.9239 - loss: 0.6054 - val_accuracy: 0.9288 - val_loss: 0.6785\n",
      "Epoch 6/10\n",
      "\u001b[1m17/17\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m27s\u001b[0m 2s/step - accuracy: 0.9273 - loss: 0.5962 - val_accuracy: 0.9328 - val_loss: 0.6807\n"
     ]
    }
   ],
   "source": [
    "history = model.fit(x=[x_train_new, y_train_new[:, :-1]],\n",
    "                    y=y_train_new[:, 1:],  \n",
    "                    batch_size=batch_size,\n",
    "                    epochs=10,\n",
    "                    validation_data=([x_test_new, y_test_new[:, :-1]], y_test_new[:, 1:]),\n",
    "                    callbacks=callbacks)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "98585c2f-f76b-4d5b-9e07-2b06250bf1f1",
   "metadata": {},
   "source": [
    "### Saving Model Weights"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 102,
   "id": "ca8e9778-29d3-4123-ab13-1dd3cced28af",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model weights saved successfully.\n"
     ]
    }
   ],
   "source": [
    "weights_file_path = 'model_weights_main.weights.h5'\n",
    "model.save_weights(weights_file_path)\n",
    "print(\"Model weights saved successfully.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "18312531-25ea-4fcc-a965-e82c808b7534",
   "metadata": {},
   "source": [
    "### Plotting model accuracy and loss"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 103,
   "id": "ebd62cc9-465a-487c-8718-513eb294531d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAjcAAAHFCAYAAAAOmtghAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjguMCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy81sbWrAAAACXBIWXMAAA9hAAAPYQGoP6dpAABJtUlEQVR4nO3deXxU9b3/8feZSTLZExLIwh52WUVQCLKKgtBypeq91lqE1tZiAUv5cfUC7l2wtVb0olgel0VqFWuDlD5QFBUCCLSgoFgRA7IJiSEsCUnINnN+f4QZMmRPJjmTyev5eJwHmTPfc+Zzxvv7zbvn+/2er2GapikAAIAAYbO6AAAAAF8i3AAAgIBCuAEAAAGFcAMAAAIK4QYAAAQUwg0AAAgohBsAABBQCDcAACCgEG4AAEBAIdwAqJPVq1fLMAwZhqGtW7dWet80TfXo0UOGYWjs2LE+/WzDMPTEE0/U+7hjx47JMAytXr3aJ+0AtAyEGwD1EhUVpRUrVlTan56eriNHjigqKsqCqgDgCsINgHq56667lJaWpry8PK/9K1asUGpqqjp37mxRZQBQjnADoF7uvvtuSdLrr7/u2Zebm6u0tDT9+Mc/rvKYc+fO6ec//7k6dOigkJAQdevWTYsWLVJxcbFXu7y8PP30pz9VfHy8IiMjdeutt+qrr76q8pwZGRn6wQ9+oISEBDkcDl1zzTV68cUXfXSV5Xbs2KHx48crKipK4eHhGjFihDZu3OjVprCwUPPnz1dKSopCQ0MVFxenoUOHen0/X3/9tb7//e+rffv2cjgcSkxM1Pjx47V//36f1gugXJDVBQBoWaKjo3XnnXdq5cqV+tnPfiapPOjYbDbdddddWrJkiVf7oqIijRs3TkeOHNGTTz6pgQMHavv27Vq8eLH279/vCQumaWrq1KnauXOnHnvsMV1//fX66KOPNGnSpEo1fPHFFxoxYoQ6d+6sZ599VklJSXr33Xf14IMPKicnR48//nijrzM9PV233HKLBg4cqBUrVsjhcOill17SlClT9Prrr+uuu+6SJM2bN09//vOf9etf/1qDBw9WQUGBPv/8c509e9ZzrsmTJ8vpdOr3v/+9OnfurJycHO3cuVMXLlxodJ0AqmACQB2sWrXKlGTu2bPH3LJliynJ/Pzzz03TNM3rr7/enDFjhmmaptmvXz9zzJgxnuNefvllU5L517/+1et8v/vd70xJ5nvvvWeapmm+8847piTz+eef92r3m9/8xpRkPv744559EydONDt27Gjm5uZ6tZ09e7YZGhpqnjt3zjRN0zx69KgpyVy1alWN11ZVu+HDh5sJCQnmxYsXPfvKysrM/v37mx07djRdLpdpmqbZv39/c+rUqdWeOycnx5RkLlmypMYaAPgO3VIA6m3MmDHq3r27Vq5cqQMHDmjPnj3Vdkl9+OGHioiI0J133um1f8aMGZKkDz74QJK0ZcsWSdI999zj1e4HP/iB1+uioiJ98MEH+t73vqfw8HCVlZV5tsmTJ6uoqEi7d+9u1PUVFBTon//8p+68805FRkZ69tvtdk2bNk3ffPONDh06JEm64YYb9M477+h//ud/tHXrVl26dMnrXHFxcerevbueeeYZ/fGPf9S+ffvkcrkaVR+AmhFuANSbYRj60Y9+pFdffVUvv/yyevXqpVGjRlXZ9uzZs0pKSpJhGF77ExISFBQU5Om+OXv2rIKCghQfH+/VLikpqdL5ysrK9L//+78KDg722iZPnixJysnJadT1nT9/XqZpKjk5udJ77du399QhSS+88IIefvhhrV+/XuPGjVNcXJymTp2qjIwMSeXf1QcffKCJEyfq97//va677jq1a9dODz74oC5evNioOgFUjXADoEFmzJihnJwcvfzyy/rRj35Ubbv4+Hh9++23Mk3Ta392drbKysrUtm1bT7uysjKvsSqSlJWV5fW6TZs2stvtmjFjhvbs2VPl5g45DdWmTRvZbDZlZmZWeu/06dOS5Kk7IiJCTz75pL788ktlZWVp2bJl2r17t6ZMmeI5pkuXLlqxYoWysrJ06NAh/fKXv9RLL72k//7v/25UnQCqRrgB0CAdOnTQf//3f2vKlCmaPn16te3Gjx+v/Px8rV+/3mv/mjVrPO9L0rhx4yRJf/nLX7zavfbaa16vw8PDNW7cOO3bt08DBw7U0KFDK21X3/2pr4iICA0bNkzr1q3z6mZyuVx69dVX1bFjR/Xq1avScYmJiZoxY4buvvtuHTp0SIWFhZXa9OrVS4888ogGDBigTz75pFF1Aqgas6UANNjTTz9da5t7771XL774oqZPn65jx45pwIAB2rFjh377299q8uTJuvnmmyVJEyZM0OjRo/XQQw+poKBAQ4cO1UcffaQ///nPlc75/PPPa+TIkRo1apQeeOABde3aVRcvXtThw4f1j3/8Qx9++GGjr23x4sW65ZZbNG7cOM2fP18hISF66aWX9Pnnn+v111/3dLMNGzZM3/3udzVw4EC1adNGBw8e1J///GelpqYqPDxcn332mWbPnq3//M//VM+ePRUSEqIPP/xQn332mf7nf/6n0XUCqIxwA6BJhYaGasuWLVq0aJGeeeYZnTlzRh06dND8+fO9pmzbbDZt2LBB8+bN0+9//3uVlJToxhtv1Ntvv60+ffp4nbNv37765JNP9Ktf/UqPPPKIsrOzFRsbq549eza6S8ptzJgx+vDDD/X4449rxowZcrlcGjRokDZs2KDvfve7nnY33XSTNmzYoOeee06FhYXq0KGD7r33Xi1atEhS+Zih7t2766WXXtLJkydlGIa6deumZ599VnPmzPFJrQC8GebVHeEAAAAtGGNuAABAQCHcAACAgEK4AQAAAYVwAwAAAgrhBgAABBTCDQAACCit7jk3LpdLp0+fVlRUVKW1bgAAgH8yTVMXL15U+/btZbPVfG+m1YWb06dPq1OnTlaXAQAAGuDkyZPq2LFjjW1aXbiJioqSVP7lREdHW1wNAACoi7y8PHXq1MnzO16TVhdu3F1R0dHRhBsAAFqYugwpYUAxAAAIKIQbAAAQUAg3AAAgoLS6MTd15XQ6VVpaanUZ8IHg4GDZ7XarywAANBNLw82yZcu0bNkyHTt2TJLUr18/PfbYY5o0aVKV7bdu3apx48ZV2n/w4EH16dPHJzWZpqmsrCxduHDBJ+eDf4iNjVVSUhLPNgKAVsDScNOxY0c9/fTT6tGjhyTplVde0W233aZ9+/apX79+1R536NAhr5lO7dq181lN7mCTkJCg8PBwfgxbONM0VVhYqOzsbElScnKyxRUBAJqapeFmypQpXq9/85vfaNmyZdq9e3eN4SYhIUGxsbE+r8fpdHqCTXx8vM/PD2uEhYVJkrKzs5WQkEAXFQAEOL8ZUOx0OrV27VoVFBQoNTW1xraDBw9WcnKyxo8fry1bttTYtri4WHl5eV5bddxjbMLDw+t/AfBr7v+mjKMCgMBnebg5cOCAIiMj5XA4NHPmTL311lvq27dvlW2Tk5O1fPlypaWlad26derdu7fGjx+vbdu2VXv+xYsXKyYmxrPVZekFuqICD/9NAaD1MEzTNK0soKSkRCdOnNCFCxeUlpam//u//1N6enq1AedqU6ZMkWEY2rBhQ5XvFxcXq7i42PPa/fjm3NzcSk8oLioq0tGjR5WSkqLQ0NCGXxT8Dv9tAaBly8vLU0xMTJW/31ez/M5NSEiIevTooaFDh2rx4sUaNGiQnn/++TofP3z4cGVkZFT7vsPh8Cy1wJIL9TN27FjNnTvX6jIAAKgXv3vOjWmaXndaarNv375WPwOmti6X6dOna/Xq1fU+77p16xQcHNzAqgAAsIal4WbhwoWaNGmSOnXqpIsXL2rt2rXaunWrNm3aJElasGCBTp06pTVr1kiSlixZoq5du6pfv34qKSnRq6++qrS0NKWlpVl5GR5Ol0vFZS6FhzTv15qZmen5+4033tBjjz2mQ4cOefa5Zwu5lZaW1im0xMXF+a5IAACaiaXdUt9++62mTZvmGRj8z3/+U5s2bdItt9wiqfxH+8SJE572JSUlmj9/vgYOHKhRo0Zpx44d2rhxo26//XarLsGjsKRMX5y+qGM5hWruYUxJSUmeLSYmRoZheF4XFRUpNjZWf/3rXzV27FiFhobq1Vdf1dmzZ3X33XerY8eOCg8P14ABA/T66697nffqbqmuXbvqt7/9rX784x8rKipKnTt31vLly5v1WgEAqI2ld25WrFhR4/tXd6U89NBDeuihh5qwospM09SlUmet7VymqeIyp1ymqXMFxQrzwd2bsGC7z2b5PPzww3r22We1atUqORwOFRUVaciQIXr44YcVHR2tjRs3atq0aerWrZuGDRtW7XmeffZZ/epXv9LChQv1t7/9TQ888IBGjx7tsydEAwDQWH435sbfXCp1qu9j71ry2V88NdFnXVxz586tdIdr/vz5nr/nzJmjTZs26c0336wx3EyePFk///nPJZUHpueee05bt24l3AAA/AbhppUYOnSo12un06mnn35ab7zxhk6dOuWZMh8REVHjeQYOHOj529395V7aAAAAf0C4qUVYsF1fPDWxTm2LS53KyM6XYRjqkxQlu61xXUphwb5bJuDq0PLss8/queee05IlSzRgwABFRERo7ty5KikpqfE8Vw9ENgxDLpfLZ3UCANBYhJtaGIZR566hsGC7osOCVVLmkmmq2WdN1cf27dt122236Yc//KEkyeVyKSMjQ9dcc43FlQEA0DiWP8QvkBiGoShHeaC5WFxmcTU169GjhzZv3qydO3fq4MGD+tnPfqasrCyrywIAoNEINz4WGVrebZNf5N8LND766KO67rrrNHHiRI0dO1ZJSUmaOnWq1WUBANBolq8t1dxqWpvCF+sPOV0ufXH6okyZ6p0UJUeQ78bNoOFYWwoAWrYWtbZUoLHbbAp3lAea/CL/7poCACAQEW6agGfcDeEGAIBmR7hpApGh5eEmv7hMrtbV6wcAgOUIN00gLNiuIJtNLtNUYUntSzcAAADfIdw0AcMwrty98fNZUwAABBrCTRNh3A0AANYg3DQR952bS6VOlTpZngAAgOZCuGkiwXabZ22ofD9/WjEAAIGEcNOEroy7IdwAANBcCDdNKMpRvhTDxaIy+fuDoMeOHau5c+d6Xnft2lVLliyp8RjDMLR+/fpGf7avzgMAgES4aVLhDrtshqEyl0tFpU03JXzKlCm6+eabq3xv165dMgxDn3zySb3OuWfPHt1///2+KM/jiSee0LXXXltpf2ZmpiZNmuTTzwIAtF6EmyZkMwxFNsOsqfvuu08ffvihjh8/Xum9lStX6tprr9V1111Xr3O2a9dO4eHhviqxRklJSXI4HM3yWQCAwEe4aWLucTcXm3BQ8Xe/+10lJCRo9erVXvsLCwv1xhtvaOrUqbr77rvVsWNHhYeHa8CAAXr99ddrPOfV3VIZGRkaPXq0QkND1bdvX23evLnSMQ8//LB69eql8PBwdevWTY8++qhKS8uf87N69Wo9+eST+vTTT2UYhgzD8NR7dbfUgQMHdNNNNyksLEzx8fG6//77lZ+f73l/xowZmjp1qv7whz8oOTlZ8fHxmjVrluezAACtW5DVBfg905RKCxt8eJThlFFaqEulhpxRkt1m1P3g4HDJqL19UFCQ7r33Xq1evVqPPfaYjMvHvPnmmyopKdFPfvITvf7663r44YcVHR2tjRs3atq0aerWrZuGDRtW6/ldLpduv/12tW3bVrt371ZeXp7X+BzPtUZFafXq1Wrfvr0OHDign/70p4qKitJDDz2ku+66S59//rk2bdqk999/X5IUExNT6RyFhYW69dZbNXz4cO3Zs0fZ2dn6yU9+otmzZ3uFty1btig5OVlbtmzR4cOHddddd+naa6/VT3/601qvBwAQ2Ag3tSktlH7bvsGHOyQNaOjBC09LIRF1avrjH/9YzzzzjLZu3apx48ZJKu+Suv3229WhQwfNnz/f03bOnDnatGmT3nzzzTqFm/fff18HDx7UsWPH1LFjR0nSb3/720rjZB555BHP3127dtX/+3//T2+88YYeeughhYWFKTIyUkFBQUpKSqr2s/7yl7/o0qVLWrNmjSIiyq996dKlmjJlin73u98pMTFRktSmTRstXbpUdrtdffr00Xe+8x198MEHhBsAAOEmUPTp00cjRozQypUrNW7cOB05ckTbt2/Xe++9J6fTqaefflpvvPGGTp06peLiYhUXF3vCQ20OHjyozp07e4KNJKWmplZq97e//U1LlizR4cOHlZ+fr7KyMkVHR9frOg4ePKhBgwZ51XbjjTfK5XLp0KFDnnDTr18/2e12T5vk5GQdOHCgXp8FAAhMhJvaBIeX30FphLxLpTp+rlAhQTb1Toyq32fXw3333afZs2frxRdf1KpVq9SlSxeNHz9ezzzzjJ577jktWbJEAwYMUEREhObOnauSkpI6nbeqaezGVd1lu3fv1ve//309+eSTmjhxomJiYrR27Vo9++yz9boG0zQrnbuqzwwODq70nsvFk6ABAISb2hlGnbuGqhMRZEoXDRXLVLERKkewvfaDGuC//uu/9Itf/EKvvfaaXnnlFf30pz+VYRjavn27brvtNv3whz+UVD6GJiMjQ9dcc02dztu3b1+dOHFCp0+fVvv25V10u3bt8mrz0UcfqUuXLlq0aJFn39Wzt0JCQuR01jwlvm/fvnrllVdUUFDguXvz0UcfyWazqVevXnWqFwDQujFbqhnYbYbCHeWBpilnTUVGRuquu+7SwoULdfr0ac2YMUOS1KNHD23evFk7d+7UwYMH9bOf/UxZWVl1Pu/NN9+s3r17695779Wnn36q7du3e4UY92ecOHFCa9eu1ZEjR/TCCy/orbfe8mrTtWtXHT16VPv371dOTo6Ki4srfdY999yj0NBQTZ8+XZ9//rm2bNmiOXPmaNq0aZ4uKQAAakK4aSbuVcKbeimG++67T+fPn9fNN9+szp07S5IeffRRXXfddZo4caLGjh2rpKQkTZ06tc7ntNlseuutt1RcXKwbbrhBP/nJT/Sb3/zGq81tt92mX/7yl5o9e7auvfZa7dy5U48++qhXmzvuuEO33nqrxo0bp3bt2lU5HT08PFzvvvuuzp07p+uvv1533nmnxo8fr6VLl9b/ywAAtEqG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NW3aNGVmZiomJkYDBw7Upk2bdMstt1hdGlBv33zzje6++27l5OSoXbt2Gj58uHbv3m3Z7yDPuQEAAAGFMTcAACCgEG4AAEBAIdwAAICAQrgBAAABhXADAAACCuEGAAAEFMINAAAIKIQbAFD54rfr16+3ugwAPkC4AWC5GTNmyDCMStutt95qdWkAWiCWXwDgF2699VatWrXKa5/D4bCoGgAtGXduAPgFh8OhpKQkr61NmzaSyruMli1bpkmTJiksLEwpKSl68803vY4/cOCAbrrpJoWFhSk+Pl7333+/8vPzvdqsXLlS/fr1k8PhUHJysmbPnu31fk5Ojr73ve8pPDxcPXv21IYNG5r2ogE0CcINgBbh0Ucf1R133KFPP/1UP/zhD3X33Xfr4MGDkqTCwkLdeuutatOmjfbs2aM333xT77//vld4WbZsmWbNmqX7779fBw4c0IYNG9SjRw+vz3jyySf1X//1X/rss880efJk3XPPPTp37lyzXicAHzABwGLTp0837Xa7GRER4bU99dRTpmmapiRz5syZXscMGzbMfOCBB0zTNM3ly5ebbdq0MfPz8z3vb9y40bTZbGZWVpZpmqbZvn17c9GiRdXWIMl85JFHPK/z8/NNwzDMd955x2fXCaB5MOYGgF8YN26cli1b5rUvLi7O83dqaqrXe6mpqdq/f78k6eDBgxo0aJAiIiI87994441yuVw6dOiQDMPQ6dOnNX78+BprGDhwoOfviIgIRUVFKTs7u6GXBMAihBsAfiEiIqJSN1FtDMOQJJmm6fm7qjZhYWF1Ol9wcHClY10uV71qAmA9xtwAaBF2795d6XWfPn0kSX379tX+/ftVUFDgef+jjz6SzWZTr169FBUVpa5du+qDDz5o1poBWIM7NwD8QnFxsbKysrz2BQUFqW3btpKkN998U0OHDtXIkSP1l7/8Rf/617+0YsUKSdI999yjxx9/XNOnT9cTTzyhM2fOaM6cOZo2bZoSExMlSU888YRmzpyphIQETZo0SRcvXtRHH32kOXPmNO+FAmhyhBsAfmHTpk1KTk722te7d299+eWXkspnMq1du1Y///nPlZSUpL/85S/q27evJCk8PFzvvvuufvGLX+j6669XeHi47rjjDv3xj3/0nGv69OkqKirSc889p/nz56tt27a68847m+8CATQbwzRN0+oiAKAmhmHorbfe0tSpU60uBUALwJgbAAAQUAg3AAAgoDDmBoDfo/ccQH1w5wYAAAQUwg0AAAgohBsAABBQCDcAACCgEG4AAEBAIdwAAICAQrgBAAABhXADAAACCuEGAAAElP8Pq7dbrtkod6cAAAAASUVORK5CYII=",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.plot(history.history['loss'])\n",
    "plt.plot(history.history['val_loss'])\n",
    "plt.title('Model loss')\n",
    "plt.ylabel('Loss')\n",
    "plt.xlabel('Epoch')\n",
    "plt.legend(['Train', 'Validation'], loc='upper left')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 104,
   "id": "6fe2217b-5cca-4089-af28-fc86dc10e8e5",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.plot(history.history['accuracy'])\n",
    "plt.plot(history.history['val_accuracy'])\n",
    "plt.title('Model Accuracy')\n",
    "plt.ylabel('Accuracy')\n",
    "plt.xlabel('Epoch')\n",
    "plt.legend(['Train', 'Validation'], loc='upper left')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 110,
   "id": "55671e56-da5b-4c9e-918c-cd00a88f5f9e",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "maximum accuracy =  92.891 %\n"
     ]
    }
   ],
   "source": [
    "print(\"maximum accuracy = \", round(max(history.history['accuracy']),5)*100, \"%\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 111,
   "id": "6ce324b5-4c12-4d9a-afc6-a941348bba30",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "maximum validation accuracy =  93.279 %\n"
     ]
    }
   ],
   "source": [
    "print(\"maximum validation accuracy = \", round(max(history.history['val_accuracy']),5)*100, \"%\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "246053c4-5a13-4d8a-bfc5-6e97e8e4377c",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
