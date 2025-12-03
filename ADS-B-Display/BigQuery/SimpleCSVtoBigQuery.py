'''
Make sure the following is installed:
pip install google-cloud-bigquery
pip install --upgrade google-api-python-client
'''

import os
import winsound
import sys
from google.cloud import bigquery

print(sys.argv)
def read_csv_file(filename):

            try:
                print(f"Read file: {filename}")
                file_path = os.path.join(global_filepath, filename) if not global_filepath.endswith(os.sep) else global_filepath + filename
                with open(file_path, "rb") as source_file:
                    job = client.load_table_from_file(
                        source_file, 
                        table_id, 
                        job_config=job_config
                    )
                job.result()  # Waits for the job to complete.
                # Delete file
                if os.path.exists(file_path):
                   os.remove(file_path)
                   print(f"File '{file_path}' deleted successfully.")
                return 0
            except Exception as e:
                print(f"Error reading file {filename}: {e}")
                return 1

if len(sys.argv) == 3:  
   global_filepath = sys.argv[1]
   filename = sys.argv[2]
   print(f"The first argument is: {global_filepath}")
   print(f"The second argument is: {filename}")
else:
   print(f"Failure 1\n")	
   os._exit(0)
current_directory = os.getcwd()
print(current_directory)
# Set credentials
api_key = os.path.join(global_filepath, "your_json_file_name")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = api_key

# Construct a BigQuery client object.
client = bigquery.Client()

# Set table_id to the ID of the table.
table_id = "your_table_id"

job_config = bigquery.LoadJobConfig(
    source_format=bigquery.SourceFormat.CSV,
    autodetect=True,
    skip_leading_rows=1,
    write_disposition=bigquery.WriteDisposition.WRITE_APPEND,   #add this line to append rows vice creating a new table or overwriting data

)
result=read_csv_file(filename)
if result == 0:
  print(f"Success\n")
  frequency = 2500  # Set Frequency (Hz)
  duration = 1000 # Set Duration (ms)
  winsound.Beep(frequency, duration)
else:
  print(f"Failure 2\n")
