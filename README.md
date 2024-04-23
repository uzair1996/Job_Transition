#    JOB TRANSITION PATHWAY: Navigating Career Transitions


This innovative system empowers job seekers by charting a clear and personalized job transition pathway tailored to their unique skill set. By analyzing an individual's current competencies, it intelligently forecasts the trajectory of their career, providing insightful guidance on both potential future roles and the skills necessary to achieve them. With this tool, job seekers can navigate the complexities of the job market with confidence, focusing on desired roles and understanding the skills they need to succeed and advance in their chosen field.


## Directory Structure


* Original_data - this folder contains original csv files of skills and job.
* Job_data - this folder contains all the jobs title and its information indeed.
* Courses_data -  this folder contains course data that has been extracted from udemy.
* Synthetic_data_cleaned_v5.csv - this file contains the data which the models required to train.
* Skills_df_updated.csv- this file contains the skills and the course link.
* Job_progression_dictionary.json - this json file contains all the jobs progression related to data science, AI.  
* Course_cleaned.csv-  this file contains the cleaned course csv file.
* All_embeddings_removed_random_jobs_only.npz/ all_embeddings_skills_only.npz/all_embeddings_job_only.npz - these are the embedding files.
* Final_job_progression_prediction.py- this is the final python runnable file which can be found inside the model folder.




 


## How to Run
Download all the files and run the final_job_progression_prediction.py, before running please download the dataset in your local machine or  which is provided below and change all the file paths.




## Recommendations Output


The final output of our sophisticated career progression tool is meticulously structured into a user-friendly table, providing a comprehensive overview of the career path possibilities tailored to the job seeker's aspirations. Here's what users can expect:


* Predicted Job Titles: A list of potential future job positions that align with the user's skill set and career goals.
* Predicted Skills: A curated set of skills that the user might need to acquire to be well-prepared for the anticipated job roles.
* Course Category Overview: The top 10 recommended course categories that have been algorithmically matched to support the user's career trajectory and skill enhancement needs.
* Top Course Recommendations: A selection of the top 10 specifically tailored courses designed to bridge any skill gaps and propel the job seeker towards their desired career outcomes.










## Note


* Make sure to install required python packages (requirements.txt)


Dataset link:


https://drive.google.com/drive/folders/1oXByJVUBO4Lp_eX3Zq7UL4fXw-aT57WD?usp=sharing






## Team contribution
# For the team contibution's table format, follow the below link
/Users/deekshitasavanur/Downloads/job_transition_pathway-main 2/TeamContribution.png

                               
Data Collection : 
        Deekshita : Jobs Data Extraction
        Gouri : Skills Data Extraction
        Uzair : Course Data Extraction
Data Cleaning : All teammates were involved
Data Processing : All teammates were involved
Data Generation & EDA : All teammates were involved
Data Modeling : All teammates were involved
Model Evaluation : All teammates were involved
Output Formatting : All teammates were involved