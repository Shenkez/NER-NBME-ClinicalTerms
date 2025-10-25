import gradio as gr
import pandas as pd
import json
import plotly.express as px
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

dataLOINC = pd.read_csv("LoincTableCore.csv", low_memory=False)
#dataLOINC = pd.read_csv(f'LoincTableCore.csv')
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
        loadPath = os.path.join(basedir, filename)  # ✅ portable and safe
        #loadPath = basedir + "\\" + filename
        #print("Loading: " + loadPath)
        return loadPath

def group_by_entity(raw):
    #outputFile = get_base(get_today_filename())
    outputFile = get_base("MedNER_results.csv")  # overwrite same file each time
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

'''
def plot_to_figure(grouped):
    fig = plt.figure()
    plt.bar(x=list(grouped.keys()), height=list(grouped.values()))
    plt.margins(0.2)
    plt.subplots_adjust(bottom=0.4)
    plt.xticks(rotation=90)
    return fig
'''
def plot_to_figure(grouped):
    if not grouped:
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.update_layout(
            title="No Entities Detected",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white")
        )
        return fig

    df = pd.DataFrame({
        "Entity Type": list(grouped.keys()),
        "Count": list(grouped.values())
    })
    fig = px.bar(
        df,
        x="Entity Type",
        y="Count",
        color="Entity Type",
        title="Entity Frequency Summary",
        text="Count",
        color_discrete_sequence=px.colors.qualitative.Antique
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        template="plotly_dark",
        font=dict(size=14, color="white"),
        title_x=0.5,
        margin=dict(t=80, l=40, r=40, b=80),
        showlegend=False,
        height=500
    )
    return fig



# ---- Your NER function ----
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
    outputDataframe = pd.read_csv(outputFile)

    # Added #Group entity counts for chart
    grouped = defaultdict(int)
    for ent in raw:
        grouped[ent["entity_group"]] += 1

    fig = plot_to_figure(grouped)

    # Return 3 outputs (highlighted text, dataframe, file)
    return (ner_content, outputDataframe, outputFile, fig)


# ---- Interface ----
# Read css from external file
with open("styles.css", "r") as f:
        css = f.read()
    
demo = gr.Blocks(theme=gr.themes.Soft(), css = css)

with demo:
    # Header
    gr.HTML(
        """
        <div style='text-align: center; margin-bottom: 20px;'>
            <h1 style='font-size: 32px; color: #FFFFFF;'>
                🩺⚕️ NBME Clinical Biomedical Named Entity Recognition
            </h1>
            <p style='font-size: 16px; color: #7fffd4; max-width: 700px; margin: auto;'>
                This model extracts <b>clinical biomedical entities</b> from patient notes, 
                such as <b>signs/symptoms</b>, <b>diagnostic procedures</b>, and <b>medications</b>. 
                Enter your clinical note below and click <b>Run →</b> to visualize the extracted entities.
            </p>
        </div>
        """
    )

    # Input Row (Textbox + Run button)
    
    with gr.Row():
        input = gr.Textbox(
            label="Note Text",
            placeholder="Type or paste a clinical note here...",
            lines=4,
            scale=5,
            elem_classes = ["Clinical-note-input"]
        )
        run_button = gr.Button("▶️ Run", size="sm", scale=1, elem_classes = ["run_button"])


    # Outputs section
    with gr.Tab("Model Output"):
        gr.HTML(
            """
            <div style='margin-bottom: 10px; color:#FFFFFF;'>
                <h3>🧠 Output Explanation:</h3>
                <ul style='line-height:1.6;'>
                    <li><b>Highlighted Text:</b> Shows entities detected in the input note.</li>
                    <li><b>Data Table:</b> Lists entities with their corresponding categories.</li>
                    <li><b>Download File:</b> CSV file containing all extracted entities.</li>
                    <li><b>Visualization:</b> Bar chart summarizing detected entity types.</li>
                </ul>
            </div>
            """
        )

        output = [
            gr.HighlightedText(label="Named Entity Recognition", combine_adjacent=True),
            gr.Dataframe(label="Extracted Entities Table"),
            gr.File(label="Download CSV File"),
            gr.Plot(label="Entity Summary Visualization"),   # ✅ NEW
        ]

    # Examples
    examples = list(EXAMPLES.keys())
    gr.Examples(examples=examples, inputs=input)
    #input.change(fn=ner, inputs=input, outputs=output)

    # Link button to function
    run_button.click(fn=ner, inputs=input, outputs=output)


# ---- Launch the app ----
if __name__ == "__main__":
    demo.launch(debug = True, share = True)

