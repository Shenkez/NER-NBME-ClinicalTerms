import gradio as gr
import pandas as pd
import json
from collections import defaultdict

# Create tokenizer for biomed model
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
tokenizer = AutoTokenizer.from_pretrained("d4data/biomedical-ner-all")    # https://huggingface.co/d4data/biomedical-ner-all?text=asthma
model = AutoModelForTokenClassification.from_pretrained("d4data/biomedical-ner-all")
pipe = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# Matplotlib for entity graph
import matplotlib.pyplot as plt
plt.switch_backend("Agg")

# Load examples from JSON
import os 

# Load terminology datasets:
basedir = os.path.dirname(__file__)
#dataLOINC = pd.read_csv(basedir + "\\" + f'LoincTableCore.csv')
#dataPanels = pd.read_csv(basedir + "\\" + f'PanelsAndForms-ACW1208Labeled.csv')     
#dataSNOMED = pd.read_csv(basedir + "\\" + f'sct2_TextDefinition_Full-en_US1000124_20220901.txt',sep='\t')   
#dataOMS = pd.read_csv(basedir + "\\" + f'SnomedOMS.csv')   
#dataICD10 = pd.read_csv(basedir + "\\" + f'ICD10Diagnosis.csv') 

dataLOINC = pd.read_csv(f'LoincTableCore.csv')
dataPanels = pd.read_csv(f'PanelsAndForms-ACW1208Labeled.csv')     
dataSNOMED = pd.read_csv(f'sct2_TextDefinition_Full-en_US1000124_20220901.txt',sep='\t')   
dataOMS = pd.read_csv(f'SnomedOMS.csv')   
dataICD10 = pd.read_csv(f'ICD10Diagnosis.csv')   

dir_path = os.path.dirname(os.path.realpath(__file__))
EXAMPLES = {}
#with open(dir_path + "\\" + "examples.json", "r") as f:
with open("examples.json", "r") as f:
    example_json = json.load(f)
    EXAMPLES = {x["text"]: x["label"] for x in example_json}

def MatchLOINC(name):
    #basedir = os.path.dirname(__file__)
    pd.set_option("display.max_rows", None)
    #data = pd.read_csv(basedir + "\\" + f'LoincTableCore.csv')    
    data = dataLOINC
    swith=data.loc[data['COMPONENT'].str.contains(name, case=False, na=False)]
    return swith
    
def MatchLOINCPanelsandForms(name):
    #basedir = os.path.dirname(__file__)
    #data = pd.read_csv(basedir + "\\" + f'PanelsAndForms-ACW1208Labeled.csv')     
    data = dataPanels
    # Assessment Name:
    #swith=data.loc[data['ParentName'].str.contains(name, case=False, na=False)]
    # Assessment Question:
    swith=data.loc[data['LoincName'].str.contains(name, case=False, na=False)]
    return swith
    
def MatchSNOMED(name):
    #basedir = os.path.dirname(__file__)
    #data = pd.read_csv(basedir + "\\" + f'sct2_TextDefinition_Full-en_US1000124_20220901.txt',sep='\t')   
    data = dataSNOMED
    swith=data.loc[data['term'].str.contains(name, case=False, na=False)]
    return swith

def MatchOMS(name):
    #basedir = os.path.dirname(__file__)
    #data = pd.read_csv(basedir + "\\" + f'SnomedOMS.csv')   
    data = dataOMS
    swith=data.loc[data['SNOMED CT'].str.contains(name, case=False, na=False)]
    return swith

def MatchICD10(name):
    #basedir = os.path.dirname(__file__)
    #data = pd.read_csv(basedir + "\\" + f'ICD10Diagnosis.csv')   
    data = dataICD10
    swith=data.loc[data['Description'].str.contains(name, case=False, na=False)]
    return swith

def SaveResult(text, outputfileName):
    #try:
    basedir = os.path.dirname(__file__)
    savePath = outputfileName
    print("Saving: " + text + " to " + savePath)
    from os.path import exists
    file_exists = exists(savePath)
    if file_exists:
        with open(outputfileName, "a") as f: #append
            #for line in text:
            f.write(str(text.replace("\n","  ")))
            f.write('\n')
    else:
        with open(outputfileName, "w") as f: #write
            #for line in text:
            f.write(str(text.replace("\n","  ")))
            f.write('\n')
    #except ValueError as err:
    #    raise ValueError("File Save Error in SaveResult \n" + format_tb(err.__traceback__)[0] + err.args[0] + "\nEnd of error message.") from None

    return

def loadFile(filename):
    try:
        basedir = os.path.dirname(__file__)
        loadPath = basedir + "\\" + filename

        print("Loading: " + loadPath)

        from os.path import exists
        file_exists = exists(loadPath)

        if file_exists:
            with open(loadPath, "r") as f: #read
                contents = f.read()
                print(contents)
                return contents

    except ValueError as err:
        raise ValueError("File Save Error in SaveResult \n" + format_tb(err.__traceback__)[0] + err.args[0] + "\nEnd of error message.") from None

    return ""

def get_today_filename():
    from datetime import datetime
    date = datetime.now().strftime("%Y_%m_%d-%I.%M.%S.%p")
    #print(f"filename_{date}")  'filename_2023_01_12-03-29-22_AM'
    return f"MedNER_{date}.csv"

def get_base(filename): 
        basedir = os.path.dirname(__file__)
        loadPath = basedir + "\\" + filename
        #print("Loading: " + loadPath)
        return loadPath

def group_by_entity(raw):
    outputFile = get_base(get_today_filename())
    out = defaultdict(int)

    for ent in raw:
        out[ent["entity_group"]] += 1
        myEntityGroup = ent["entity_group"]
        print("Found entity group type: " + myEntityGroup)

        if (myEntityGroup in ['Sign_symptom', 'Detailed_description', 'History', 'Activity', 'Medication' ]):
            eterm = ent["word"].replace('#','')
            minlength = 3
            if len(eterm) > minlength:
                print("Found eterm: " + eterm)
                eterm.replace("#","")
                g1=MatchLOINC(eterm)
                g2=MatchLOINCPanelsandForms(eterm)
                g3=MatchSNOMED(eterm)
                g4=MatchOMS(eterm)
                g5=MatchICD10(eterm)
                sAll = ""

                print("Saving to output file " + outputFile)
                # Create harmonisation output format of input to output code, name, Text

                try: # 18 fields, output to labeled CSV dataset for results teaching on scored regret changes to action plan with data inputs
                    col = "          1                            2            3         4            5                     6                    7                       8                   9              10                   11                         12       13               14                      15                  16                            17                    18                       19"
                    
                    #LOINC
                    g11 = g1['LOINC_NUM'].to_string().replace(","," ").replace("\n"," ")
                    g12 = g1['COMPONENT'].to_string().replace(","," ").replace("\n"," ")
                    s1 = ("LOINC," + myEntityGroup + "," + eterm + ",questions of ," + g12 + "," + g11 + ", Label,Value, Label,Value, Label,Value  ")
                    if g11 != 'Series([]  )': SaveResult(s1, outputFile)

                    #LOINC Panels
                    g21 = g2['Loinc'].to_string().replace(","," ").replace("\n"," ")
                    g22 = g2['LoincName'].to_string().replace(","," ").replace("\n"," ")
                    g23 = g2['ParentLoinc'].to_string().replace(","," ").replace("\n"," ")
                    g24 = g2['ParentName'].to_string().replace(","," ").replace("\n"," ")
                    # s2 = ("LOINC Panel," + myEntityGroup + "," + eterm + ",name of ," + g22 + "," + g21 + ", and Parent codes of ," + g23 + ", with Parent names of ," + g24 + ", Label,Value  ")
                    s2 = ("LOINC Panel," + myEntityGroup + "," + eterm + ",name of ," + g22 + "," + g21 + "," + g24 + ", and Parent codes of ," + g23 + "," + ", Label,Value  ")
                    if g21 != 'Series([]  )': SaveResult(s2, outputFile)

                    #SNOMED
                    g31 = g3['conceptId'].to_string().replace(","," ").replace("\n"," ").replace("\l"," ").replace("\r"," ")
                    g32 = g3['term'].to_string().replace(","," ").replace("\n"," ").replace("\l"," ").replace("\r"," ")
                    s3 = ("SNOMED Concept," + myEntityGroup + "," + eterm + ",terms of ," + g32 + "," + g31 + ", Label,Value, Label,Value, Label,Value  ")
                    if g31 != 'Series([]  )': SaveResult(s3, outputFile)

                    #OMS
                    g41 = g4['Omaha Code'].to_string().replace(","," ").replace("\n"," ")
                    g42 = g4['SNOMED CT concept ID'].to_string().replace(","," ").replace("\n"," ")
                    g43 = g4['SNOMED CT'].to_string().replace(","," ").replace("\n"," ")
                    g44 = g4['PR'].to_string().replace(","," ").replace("\n"," ")
                    g45 = g4['S&S'].to_string().replace(","," ").replace("\n"," ")
                    s4 = ("OMS," + myEntityGroup + "," + eterm + ",concepts of ," + g44 + "," + g45 + ", and SNOMED codes of ," + g43 + ", and OMS problem of ," + g42 + ", and OMS Sign Symptom of ," + g41)
                    if g41 != 'Series([]  )': SaveResult(s4, outputFile)

                    #ICD10
                    g51 = g5['Code'].to_string().replace(","," ").replace("\n"," ")
                    g52 = g5['Description'].to_string().replace(","," ").replace("\n"," ")
                    s5 = ("ICD10," + myEntityGroup + "," + eterm + ",descriptions of ," + g52 + "," + g51 + ", Label,Value, Label,Value, Label,Value  ")
                    if g51 != 'Series([]  )': SaveResult(s5, outputFile)

                except ValueError as err:
                    raise ValueError("Error in group by entity \n" + format_tb(err.__traceback__)[0] + err.args[0] + "\nEnd of error message.") from None

    return outputFile


def plot_to_figure(grouped):
    fig = plt.figure()
    plt.bar(x=list(grouped.keys()), height=list(grouped.values()))
    plt.margins(0.2)
    plt.subplots_adjust(bottom=0.4)
    plt.xticks(rotation=90)
    return fig


def ner(text):
    raw = pipe(text)
    ner_content = {
        "text": text,
        "entities": [
            {
                "entity": x["entity_group"],
                "word": x["word"],
                "score": x["score"],
                "start": x["start"],
                "end": x["end"],
            }
            for x in raw
        ],
    }
    
    outputFile = group_by_entity(raw)
    label = EXAMPLES.get(text, "Unknown")
    outputDataframe = pd.read_csv(outputFile)
    return (ner_content, outputDataframe, outputFile)

demo = gr.Blocks()
with demo:
    gr.Markdown(
    """
    # 🩺⚕️NLP Clinical Ontology Biomedical NER 
    """
    )
    input = gr.Textbox(label="Note text", value="")

    with gr.Tab("Biomedical Entity Recognition"):
        output=[
            gr.HighlightedText(label="NER", combine_adjacent=True),
            #gr.JSON(label="Entity Counts"),
            #gr.Label(label="Rating"),
            #gr.Plot(label="Bar"),
            gr.Dataframe(label="Dataframe"),
            gr.File(label="File"),
        ]
        examples=list(EXAMPLES.keys())  
        gr.Examples(examples, inputs=input)
        input.change(fn=ner, inputs=input, outputs=output)
        
    with gr.Tab("Clinical Terminology Resolution"):
        with gr.Row(variant="compact"):
            btnLOINC = gr.Button("LOINC")
            btnPanels = gr.Button("Panels")
            btnSNOMED = gr.Button("SNOMED")
            btnOMS = gr.Button("OMS")
            btnICD10 = gr.Button("ICD10")

        examples=list(EXAMPLES.keys())  
        gr.Examples(examples, inputs=input)
        input.change(fn=ner, inputs=input, outputs=output)
#layout="vertical"
demo.launch(debug=True)
