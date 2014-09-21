import os, json
import urllib

def contains_subject(subject, list):
	tokens = subject.split(" ")
	for s in tokens:
		if (s == "and"):
			continue
		for l in list:
			if(s in l.lower()):
				return True
	return False	

def read_json(reps, csv):
	committee_codes = {"SPAG":1, "SSAF":2, "SSAP":3, "SSAS":4, "SSBK":5, "SSBU":6, "SSCM":7, "SSEG":8, "SSEV":9, "":10, "SLET":11, "SSFI":12, "SSFR":13, "SSFR":14, "SSGA":15, "SLIA":16, "SLIN":17, "SSJU":18, "SSRA":19, "SSSB":20, "SCNC":21, "SSVA":22, "HSAG":23, "HSAP":24, "HSAS":25, "HSBU":26, "HSED":27, "HSIF":28, "HSSO":29, "HLZI":30, "HSBA":31, "HSFA":32, "HSHM":33, "HSHA":34, "HLIG":35, "HSJU":36, "HSII":37, "HSGO":38, "HSRU":39, "HSSY":40, "HSSM":41, "HSPW":42, "HSVR":43, "HSWM":44 }
	bill_types = {"hr": "house_bill", "s": "senate_bill", "hjres": "house_resolution", "sjres":"senate_resolution"}
	try:
		f = open("data.json")
	except:
		print "COULD NOT OPEN FILE in ", os.getcwd()
		return
	j = json.load(f)
	
	# Load the current status. ENACTED:SIGNED = 1, REFERRED = 0. Otherwise return.
	if j['status'] == "ENACTED:SIGNED":
		status = 1
	elif j['status'] == "REFERRED":
		status = 0
	else:
		return
	# Check if the bill has been referred to a committee.
	if "committees" in j and j["committees"] != []:
		if j["committees"][0]["committee_id"] in committee_codes:
			committees = committee_codes[j["committees"][0]["committee_id"]]
		else:
			committees = 0
	else:
		committees = 0
	# Get the congress that the bill was in.
	congress = j['congress']
	# Get the sponsor district.
	if "sponsor" in j and j["sponsor"] != None:
		sponsor_district = j["sponsor"]["district"]
	else:
		sponsor_district = 0
	# Date introduced.
	date_introduced = j["introduced_at"]
	
	if(date_introduced == None):
		return
		
	num_cosponsors = len(j["cosponsors"])
	num_co_dem = 0
	if num_cosponsors > 0:
		for co in j["cosponsors"]:
			if co["thomas_id"] in reps:
				party = reps[co["thomas_id"]][6]
				if party == "Democrat":
					num_co_dem += 1
				
	#Get Bill detailed info
	if j["sponsor"] != None and j["sponsor"]["thomas_id"] != None and j["sponsor"]["thomas_id"] in reps:
		sponsor_party = reps[j["sponsor"]["thomas_id"]][6]
		sponsor_gender = reps[j["sponsor"]["thomas_id"]][3]
		sponsor_role = reps[j["sponsor"]["thomas_id"]][4]
	else:
		sponsor_party = 0
		sponsor_gender = 0
		sponsor_role = 0
	
	#check if dominant subjects are present
	csv.write(str(congress) + "," + str(sponsor_district) + "," + str(committees) + "," + str(date_introduced) + "," + str(sponsor_party) + "," + str(sponsor_gender) + "," + str(sponsor_role) + "," + str(num_cosponsors) + "," + str(num_co_dem) + "," + str(status) + "\n")

base_dir = os.getcwd()
bill_ids = {}
reps = {}
status = {}
url_data = urllib.urlopen("https://www.govtrack.us/data/congress-legislators/legislators-historic.csv").read()
url_data_curr = urllib.urlopen("https://www.govtrack.us/data/congress-legislators/legislators-current.csv").read()
person_info_arr = url_data.split("\n")
for p in person_info_arr:
	info = p.split(",")
	if(len(info) > 18):
		reps[info[18]] = info
person_info_arr = url_data_curr.split("\n")
for p in person_info_arr:
	info = p.split(",")
	if(len(info) > 18):
		reps[info[18]] = info
csv = open("/home/mike/Desktop/comp598-ass1/data_after107.csv", "w")
csv.write("Congress,Sponsor District,Committees,Date Introduced, Sponsor Party, Sponsor Gender, Sponsor Role, Cosponsors, "+
			"Number of Democratic Cosponsors," +
			"Status\n")

for congress in xrange(90, 114):
	congress = str(congress)
	os.chdir(congress)
	try:
		os.chdir("bills")
	except OSError:
		os.chdir(base_dir)
		continue
	for bill_type in ["hjres", "hr", "s", "sjres"]:
		print congress, bill_type
		try:
			os.chdir(bill_type)
		except OSError:
			continue
		for bill in os.listdir(os.getcwd()):
			try:
				os.chdir(bill)
			except OSError:
				continue
			read_json(reps, csv)
			os.chdir(base_dir + "/" + congress + "/bills/" + bill_type)
		os.chdir(base_dir + "/" + congress + "/bills")
	os.chdir(base_dir)
csv.close()
	
