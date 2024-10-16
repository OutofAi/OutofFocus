
<div align="center">

![image](https://github.com/user-attachments/assets/55a38e74-ab93-4d80-91c8-0fa6130af45a)

<h1>Out of Focus v1.0</h1>

<a href="https://www.buymeacoffee.com/outofai" target="_blank"><img src="https://img.shields.io/badge/-buy_me_a%C2%A0coffee-red?logo=buy-me-a-coffee" alt="Buy Me A Coffee"></a>
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=Ashleigh%20Watson)](https://twitter.com/OutofAi) 
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=Alex%20Nasa)](https://twitter.com/banterless_ai)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/OutofAi/OutofFocus/blob/main/app_turbo_colab.ipynb)


</div>

| **SD2.1 (15 GB) uses pivotal tunining, slow editing** | **SD1.5 (8GB) uses latent alignment, fast editing** |
|:----------------------:|:----------------------:|
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/OutofAi/OutofFocus/blob/main/app_colab.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/OutofAi/OutofFocus/blob/main/app_turbo_colab.ipynb) |


Out of AI presents a flexible tool in Gradio to manipulate your images. This is our first version of Image modification tool through prompt manipulation by reconstruction through diffusion inversion process.


https://github.com/user-attachments/assets/de8a4f9d-cdad-4b73-962e-2cc42bbc2e0d



## Download the project

```bash
git clone https://github.com/OutofAi/OutofFocus.git
cd  OutofFocus
```

# 🚀 Run Gradio Demo locally
After downloading the project and navigating to the correct folder

**Install dependencies:**
```bash
pip install -r requirements.txt
```
**Run the app**
```bash
python app.py
```
If you running on a GPU with less than 15GB VRAM, try running the turbo variation as it only requires 8 GB VRAM, this model
uses SD 1.5 and can handle any aspect ratios as well
**Run the app**
```bash
python app_turbo.py
```

# 🚀 Run Gradio Demo locally in Virtual Environment
If you face any dependency, token or cuda issues we higly recommend running it in a virtual environment.

To run it in a virtual environment use these commands instead, after downloading the project and navigating to the correct folder

**Create a virtual environment:**
```bash
python -m venv venv
```

**Activate the environment:**

*Windows:*
```bash
.\venv\Scripts\activate
```
*macOS/Linux:*
```bash
source venv/bin/activate
```
**Install dependencies:**
```bash
pip install -r requirements.txt
```
**Run the app in virtual environment:**
```bash
python app.py
```

# 🚀 Run Gradio Demo on Google Colab (GPU based)



If you want to run it on [Colab](https://colab.research.google.com/) either click on the badge below 


 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/OutofAi/OutofFocus/blob/main/app_colab.ipynb)

 or create your own notebook, make sure you got a GPU notebook enabled (Views->Notebook Info) a A100 GPU or a L4 GPU and run the following commands


```bash
!git clone https://github.com/OutofAi/OutofFocus
%cd OutofFocus
!pip install -r requirements.txt
!python app.py --share
```


https://github.com/user-attachments/assets/1f71553d-d41d-45d6-a256-df3e06f93182


# More Examples:
![image](https://github.com/user-attachments/assets/d9f7aac4-abd6-448f-9c1a-c046958086a9)
![image](https://github.com/user-attachments/assets/a19e1a43-de42-4244-ba39-5fcdbff509d4)
![image](https://github.com/user-attachments/assets/6b80e011-3959-4b3e-a686-365bdb32ae94)
![image](https://github.com/user-attachments/assets/62a324b5-a792-438a-97c5-0e40953a84ed)



