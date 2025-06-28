
<div align="center">

![8d7a77ba-52b3-427b-8d6b-6e9d570f5939-removebg-preview (2) (1)](https://github.com/user-attachments/assets/9b92a2cd-4c1f-4de2-87f0-09053fe129ff)

<h1>Out of Focus v1.0</h1>

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/OutofAi/OutofFocus/blob/main/app_turbo_colab.ipynb)

[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=alexnasa)](https://twitter.com/alexandernasa)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=Out%20of%20AI)](https://twitter.com/OutofAi)

</div>

Out of AI presents a flexible tool in Gradio to manipulate your images. This is our first version of Image modification tool through prompt manipulation by reconstruction through diffusion inversion process.

<div align="center">

https://github.com/user-attachments/assets/de8a4f9d-cdad-4b73-962e-2cc42bbc2e0d

</div>


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
python app_turbo.py
```
You can also run the SD 2.1 version that requires A100 GPU for efficient editting

**Run the app**
```bash
python app.py
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
python app_turbo.py
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


# Disclaimer
This project aims to contribute positively to the field of AI-based image generation. While users are encouraged to explore its creative capabilities, they are responsible for adhering to local regulations and using the tool responsibly. The developers are not liable for any misuse by users.

# More Examples:
![image](https://github.com/user-attachments/assets/d9f7aac4-abd6-448f-9c1a-c046958086a9)
![image](https://github.com/user-attachments/assets/a19e1a43-de42-4244-ba39-5fcdbff509d4)
![image](https://github.com/user-attachments/assets/6b80e011-3959-4b3e-a686-365bdb32ae94)
![image](https://github.com/user-attachments/assets/62a324b5-a792-438a-97c5-0e40953a84ed)



