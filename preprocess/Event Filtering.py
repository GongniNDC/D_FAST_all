import pandas as pd
import csv

# #Select noise signals near the B023 station#############
input_file_name = "E:/D_FAST_code/catalog/noise_metadata.csv"
output_file_name = "E:/D_FAST_code/catalog/noise_selected.csv"

with open(input_file_name, mode='r', newline='') as infile, \
     open(output_file_name, mode='w', newline='') as outfile:

    # reate CSV Reader and Write
    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    # Read and Write Header Row
    headers = next(reader)
    writer.writerow(headers)

    # Read Line by Line
    for row in reader:
        if 'B023' in row:
            # If contains, write that line to the output file
            writer.writerow(row)

#############Select earthquake events within the specified area############
type1='eq'
type2='ep'
# Area##
min_latitude = 46.0
max_latitude = 46.5
min_longitude = -123.5
max_longitude = -123.0
def select_events(type):
    input_file_name =f"E:/D_FAST_data/catalog/{type}.csv"
    output_file_name = f"E:/D_FAST_data/catalog/{type}_selected.csv"


    # Open input file and output file
    with open(input_file_name, mode='r', newline='') as infile, \
         open(output_file_name, mode='w', newline='') as outfile:

        # reate CSV Reader and Write
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        # Read and Write Header Row
        headers = next(reader)
        writer.writerow(headers)

        # Read Line by Line
        for row in reader:
            # Obtain latitude and longitude, and convert them to float
            latitude = float(row[2])
            longitude = float(row[3])

            # Check if the filtering criteria are met
            if min_latitude <= latitude <= max_latitude and min_longitude <= longitude <= max_longitude:
                # If the conditions are met, write that line to the output file.
                writer.writerow(row)
select_events(type1)
select_events(type2)