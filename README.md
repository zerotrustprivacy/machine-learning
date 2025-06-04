# Machine Learning Operations
## Continuous Training
 <a href="https://open.substack.com/pub/techsavvysadie/p/building-data-pipelines-in-gcp-for?r=573b3l&utm_campaign=post&utm_medium=web&showWelcomeOnShare=false" target="_blank"> Read my blog!</a>

 ![image](https://github.com/user-attachments/assets/43c0a157-0fc7-4f99-931f-0cf5bf3ebd93)
 https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning?_gl=1*1bi51db*_ga*MTkwNTUzNzY2OS4xNzQzOTY3NTcy*_ga_WH2QY8WWF5*czE3NDgyMjUzNDYkbzI1JGcxJHQxNzQ4MjI2MTE0JGo2MCRsMCRoMCRkQkZuSGQ4bGJRMFVhZXJhd3dFSkZSRktqSDhNVUFWNl9KUQ..#mlops_level_1_ml_pipeline_automation
 #
 # Real Time Data Streaming and Analysis
 ## Building a pipeline that continuously ingests simulated patient telemetry data, processes it in real-time (e.g., aggregates metrics over short intervals), and lands the aggregated data into BigQuery for immediate analysis.
<p>
 <ul>
  <p>GCP Services Used:</p>

<li>Cloud Pub/Sub </li>
<li>Cloud Dataflow (Streaming Mode)</li>
<li>BigQuery: As the real-time analytics data warehouse where your processed data will be stored.</li>
 </ul>
</p>
<p>Install Python 3 and the necessary libraries into the environment</p>
pip install apache-beam[gcp] google-cloud-pubsub google-cloud-bigquery pandas
<p>Create Pub/Sub Topic, Subscription</p>
gcloud pubsub topics create telemetry-data-stream --project YOUR_GCP_PROJECT_ID
gcloud pubsub subscriptions create telemetry-data-stream-sub \
  --topic telemetry-data-stream \
  --ack-deadline=600 \
  --message-retention-duration=7d \
  --project YOUR_GCP_PROJECT_ID
<p>Create Big Query Dataset to house tables</p>
gcloud bq mk --location=us-central1 YOUR_GCP_PROJECT_ID:telemetry_analytics
<p>Create a JSON file with the telemetry schema</p>


