# MLops Project
The objective of this project was to identify a problem to work on and implement MLOps techniques to build an end-to-end solution, from model training to deployment. 🚀

## Table of Contents 📚
- [Project Scope](#project-scope)
- [Architecture](#architecture)
- [Experiement](#experiemente)
- [User Interface](#user_interfac)
- [How to run](#how-to-run)
- [Challenges Encountered](#challenges-encountered)

## **Project Focus**

It is important to note that this project is primarily an **MLOps project** rather than a model performance-driven study. The emphasis is on:  

- Data collection and structuring
- Experiment Tracking with MLflow
- Model training pipeline automation  
- Deployment and reproducibility  

Accuracy, metrics, and model performance optimization were **not** the main objectives. Instead, the goal was to explore **workflow automation**.


## Use Case : Brain MRI Analysis 🧠

<p align="center">
   <img src="https://www.pixenli.com/image/4K-b5BX8" alt="Brainy">
</p>

**Topic**: Computer Vision  
**Field of Application**: Healthcare  
**Use Case Description** : After exploring various datasets online for this project, We decided to cross-reference different data sources to address the following research question:  
**How can we determine whether an MRI scan provides evidence of a specific disease?** 

## Architecture ️

The goal is to implement this pipeline:

<p align="center">
   <img src="https://www.pixenli.com/image/2hnZzHBv" alt="Architecture Diagram">
</p>


* **Part 1:** Aside from the resources collected from Kaggle, which were loaded locally as demonstrated in the notebook, the ideal solution would involve automating this part with Airflow, especially if we had access to a resource that gets updated regularly. In our case, we didn't automate it, but we still uploaded the data to AWS.

* **Part 2:** Once the data was collected, we cleaned and processed it to make it suitable for the model. We used MLflow to track different executions and register the best model. 

* **Part 3:** We created an API using the FastAPI framework and connected it to the interface developed with Gradio.



## Experimentation

We conducted experiments on model training and tracking using MLflow. We had two key experiments: the first focused on fine-tuning hyperparameters, and the second was dedicated to selecting and registering the best model.

### Experiment 1: Hyperparameter Tuning
In the first experiment, we explored hyperparameter tuning to optimize the model's performance. Below is a visual representation of the results:

<p align="center">
   <img src="https://www.pixenli.com/image/eOgeUBcd" alt="Experiment 1: Hyperparameter Tuning">
</p>

### Model Comparison
The following chart compares the performance of different models during the experimentation process:

<p align="center">
   <img src="https://www.pixenli.com/image/tcp-BtDJ" alt="Model Comparison">
</p>

### Experiment 2: Model Selection and Registration
In the second experiment, we selected the final model based on the best performance and registered it for future use. Below is a representation of the registered model in this experiment:

<p align="center">
   <img src="https://www.pixenli.com/image/Cv1zmo_1" alt="Registered Model">
</p>


**For inference, we ensured that the latest registered model is used. If the MLflow runs are empty, we create a separate file to load the model from Google Drive.**

## User Interface

We used Gradio to develop a simple UI that accepts an image, scans it, and makes a prediction using our model. We also incorporated model interpretability with the LIME library.

Below is a representation of our interface:

<p align="center">
   <img src="https://www.pixenli.com/image/I-7nRVhW" alt="User Interface">
</p>

And here is an example in action:

<p align="center">
   <img src="https://www.pixenli.com/image/LVJ50iwS" alt="Interface Example">
</p>

## 🚀 How to Run the Project  

#### 1️⃣ Clone the Repository  
```bash
git clone git@github.com:Biline-dev/MLops_Project.git
cd MLops_Project
```  

#### 2️⃣ Start the Application  
```bash
docker-compose up  # Builds and runs the backend and frontend containers
```  

OR 
### Manually set up (Linux/wsl)

Follow these steps to manually set up the environment for both the **backend** and **frontend** without Docker.

#### Open Two Terminals  
You'll need to run the backend and frontend separately.

---

#### Backend Setup  

#### Create and Activate a Virtual Environment  
```bash
cd backend/
python3 -m venv venv
source venv/bin/activate
```

#### Install Dependencies  
```bash
pip install -r requirements.txt
```

#### Run Backend  
```bash
python python.py
```


#### Frontend Setup  

#### Create and Activate a Virtual Environment  
Open a second terminal and run:
```bash
cd frontend/
python3 -m venv venv
source venv/bin/activate
```

#### Install Dependencies  
```bash
pip install -r requirements.txt
```

#### Run Frontend  
```bash
python interface.py
```


### ⚠️ Important Notes  
- Building the images or installing dependencies may take some time.  
- We couldn't upload the model to GitHub or deploy MLflow to retrieve the latest registered model as we did locally.  
- Instead, the model will be downloaded from Google Drive, which might take a few minutes.  



### 🌐 Access the Application  
Once the setup is complete, open your browser and go to:  

🔗 **[http://127.0.0.1:7860/](http://127.0.0.1:7860/)**  

This will allow you to test the app. 🚀  



## Next Implementation
The next steps will involve testing the orchestration part, implementing monitoring, and enabling deployments across providers like AWS or GCP—something we couldn't complete due to time and resources constraints.





