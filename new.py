import csv
import openai
import pickle
from tqdm import tqdm
import json
import random
import os
import multiprocessing as mp
import re
from os.path import join
from datetime import datetime
import time
import fcntl
import numpy as np
import hydra
from omegaconf import DictConfig
from loguru import logger
import langchain
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from thefuzz import process

from utils.nlp import calculate_num_tokens, truncate_text, create_lab_test_string
from dataset.utils import load_hadm_from_file
from utils.logging import append_to_pickle_file
from evaluators.appendicitis_evaluator import AppendicitisEvaluator
from evaluators.cholecystitis_evaluator import CholecystitisEvaluator
from evaluators.diverticulitis_evaluator import DiverticulitisEvaluator
from evaluators.pancreatitis_evaluator import PancreatitisEvaluator
from models.models import CustomLLM

def load_evaluator(pathology):
    # Load desired evaluator
    if pathology == "appendicitis":
        evaluator = AppendicitisEvaluator()
    elif pathology == "cholecystitis":
        evaluator = CholecystitisEvaluator()
    elif pathology == "diverticulitis":
        evaluator = DiverticulitisEvaluator()
    elif pathology == "pancreatitis":
        evaluator = PancreatitisEvaluator()
    else:
        raise NotImplementedError
    return evaluator


@hydra.main(config_path="./configs", config_name="config", version_base=None)
def run(args: DictConfig):
    if args.self_consistency:
        args.save_probabilities = True
    else:
        random.seed(args.seed)
        np.random.seed(args.seed)

    # Load lab test mapping
    with open(args.lab_test_mapping_path, "rb") as f:
        lab_test_mapping_df = pickle.load(f)

    # Load patient data
    for patho in ["appendicitis", "cholecystitis", "diverticulitis", "pancreatitis"]:
    # patho = args.pathology
        hadm_info_clean = load_hadm_from_file(
                f"{patho}_hadm_info_first_diag", base_mimic=args.base_mimic
        )

        # Load list of specific IDs if provided
        patient_list = hadm_info_clean.keys()
        if args.patient_list_path:
            with open(args.patient_list_path, "rb") as f:
                patient_list = pickle.load(f)

        first_patient_seen = False
        for _id in patient_list:
            if args.first_patient and not first_patient_seen:
                if _id == args.first_patient:
                    first_patient_seen = True
                else:
                    continue
            # logger.info(f"Processing patient: {_id}")
            hadm = hadm_info_clean[_id]


            # Eval
            evaluator = load_evaluator(
                args.pathology
            )  # Reload every time to ensure no state is carried over

            input = {}

            #defining a dicctionary where every character represents a function
            char_to_func = {

                "l": "include_laboratory_tests",

            }

            # Read desired order from mapping and args.order and then execute and parse result
            for char in args.order:
                func = char_to_func[char]
                # Must be within for loop to use updated input variable
                mapping_functions = {

                    "include_laboratory_tests": (
                        add_laboratory_tests,
                        [input, hadm, evaluator, lab_test_mapping_df, args],
                    ),
                } #these are the actual functions to be called based on the order string

                function, input_params = mapping_functions[func]
                result = function(*input_params)

                if isinstance(result, tuple):
                    input, rad_reports = result
                else:
                    input = result

            hadm.update(input)
            keys_to_remove = ['Reference Range Lower', 'Reference Range Upper']
            for key in keys_to_remove:
                if key in hadm:
                   del hadm[key]

            ChatGPT_data_converter(_id, hadm, patho)



def add_laboratory_tests(input, hadm, evaluator, lab_test_mapping_df, args):

    if args.include_ref_range:
        input["Laboratory Tests"]  = (
            "(<FLUID>) <TEST>: <RESULT> | REFERENCE RANGE (RR): [LOWER RR - UPPER RR]\n"
        )

    else:
        input["Laboratory Tests"] = "(<FLUID>) <TEST>: <RESULT>\n"
    lab_tests_to_include = []

    for test_name in evaluator.required_lab_tests:
        lab_tests_to_include = (
            lab_tests_to_include + evaluator.required_lab_tests[test_name]
        )
    lab_tests_to_include = lab_tests_to_include + evaluator.neutral_lab_tests

    for test in lab_tests_to_include:
        if test in hadm["Laboratory Tests"].keys():
            input["Laboratory Tests"] += create_lab_test_string(
                test,
                lab_test_mapping_df,
                hadm,
                include_ref_range=args.include_ref_range,
                bin_lab_results=args.bin_lab_results,
                bin_lab_results_abnormal=args.bin_lab_results_abnormal,
                only_abnormal_labs=args.only_abnormal_labs,
            )

    return input

def process_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        file_content = file.read()
        return file_content

def ChatGPT_data_converter(id, hadm, pathology):
    openai.api_key = ''
    # Replace with your directory path
    out_file = 'output.json'
    messages = []
    messages.append(
        {"role": "system",
        "content": "You are a helpful assistant. Carefully create questions and multiple choice answers from given medical records in a valid JSON format." }  )

    valid_count = 0
    count = 0
    invalid_list = []
    data_out = []


    message = ("""You are a medical expert. You will be provided with information about a patient suffering from an ailment. 
               Your task is to analyze the symptoms, laboratory results, and imaging findings from the medical record to generate 
               a multiple-choice question (MCQ) that asks for the most likely diagnosis. 
               The options should include the following diagnoses: appendicitis, cholecystitis, diverticulitis, and pancreatitis. 
               The correct diagnosis should be the one present in the medical record, identified in the "Discharge Diagnosis" or 
               "ICD Diagnosis" field or derived from the clinical data provided. 
               Do not explicitly state the diagnosis in the question. Include key clinical findings in your summary and question to
              guide the answer. Output the MCQ in the following JSON format:
               {
                "id":"""+ str(id) + """,
                "question": "Provide the most likely final diagnosis for the following patient. [Condensed summary of the patient's presentation, including symptoms, physical examination findings, laboratory results, and imaging findings]. What is the final diagnosis for this patient?",
                "answer_choices": {
                    "A": "Appendicitis",
                    "B": "Cholecystitis",
                    "C": "Diverticulitis",
                    "D": "Pancreatitis"
                },
                "correct_answer": """ + pathology + """
                }
        Here is an example of how an example MCQ should look. Without newline characters, backslashes, or extraneous quotes, to be in JSON format::
               {
                "id": 33677,
                "question": "Provide the most likely final diagnosis for the following patient. An otherwise healthy woman presents with periumbilical to right lower quadrant pain, nausea, and non-bloody, non-bilious vomiting. Physical examination reveals tenderness in the right lower quadrant without rebound or guarding. Laboratory results show elevated white blood cell count. CT abdomen reveals an appendicolith and an enlarged appendix with surrounding fat stranding. What is the final diagnosis for this patient?",
                "answer_choices": {
                    "A": "Appendicitis",
                    "B": "Cholecystitis",
                    "C": "Diverticulitis",
                    "D": "Pancreatitis"
                },
                "correct_answer": "(Pathology name)"
}
            Please ensure that:

            The summary is concise but includes the most relevant clinical information.For laboratory tests, the blood and urine reference ranges are provided as "RR" to the right of the observation. 
            The question does not reveal the diagnosis directly.
            The JSON output is valid and properly formatted without any unnecessary escape characters. 
            Now it is your turn to create a JSON file based on the medical record provided.""" + str(hadm)
        
        
    )

    if message:
        
        messages.append(
            {"role": "user", "content": message},
        )
        try:

            chat = openai.ChatCompletion.create(
              model="gpt-3.5-turbo", messages=messages, temperature=1
             )
            reply = chat.choices[0].message.content
            data_out.append(reply)
            valid_count += 1
        except openai.error.APIError as e:
                    # Handle API error here, e.g. retry or log
                    print("Count ID ", count)
                    invalid_list.append(count)
                    print(f"OpenAI API returned an API Error: {e}")
                    with open(out_file, 'a') as f:
                        json.dump(data_out, f, indent=2, ensure_ascii=False)
                        f.write(',')
                    pass
        except openai.error.APIConnectionError as e:
                    # Handle connection error here
                    print("Count ID ", count)
                    invalid_list.append(count)
                    print(f"Failed to connect to OpenAI API: {e}")
                    with open(out_file, 'a') as f:
                        json.dump(data_out, f, indent=2, ensure_ascii=False)
                        f.write(',')
                    pass
        except openai.error.RateLimitError as e:
                    # Handle rate limit error (we recommend using exponential backoff)
                    invalid_list.append(count)
                    print(
                        f"openai.error.RateLimitError, This file is not processed ", count)
                    with open(out_file, 'a') as f:
                        json.dump(data_out, f, indent=2, ensure_ascii=False)
                        f.write(',')
                    pass
        except openai.error.InvalidRequestError as e:
                    invalid_list.append(count)
                    print(
                       f"openai.error.InvalidRequestError, This file is not processed ", count)
                    with open(out_file, 'a') as f:
                        json.dump(data_out, f, indent=2, ensure_ascii=False)
                        f.write(',')
                    pass

        except openai.error.Timeout as e:
                    invalid_list.append(count)
                    print(
                        f"openai.error.Timeout, This file is not processed ", count)
                    with open(out_file, 'a') as f:
                        json.dump(data_out, f, indent=2, ensure_ascii=False)
                        f.write(',')
                    pass
        except openai.error.ServiceUnavailableError as e:
                    invalid_list.append(count)
                    print(
                        f"openai.error.ServiceUnavailableError, This file is not processed ", count)
                    with open(out_file, 'a') as f:
                        json.dump(data_out, f, indent=2, ensure_ascii=False)
                        f.write(',')
                    pass
        except ValueError:  
                    invalid_list.append(count)
                    print(
                        f"JSON format is incorrect, This file is not processed ", count)
                    with open(out_file, 'a') as f:
                        json.dump(data_out, f, indent=2, ensure_ascii=False)
                        f.write(',')
                    pass
        count += 1
        messages.pop()
    with open(out_file, 'a') as f:
      json.dump(data_out, f, indent=2, ensure_ascii=False)
      f.write(',')

if __name__ == "__main__":
    run()

#     You are a medical expert. You will be provided information about a patient suffering from an ailment. Your task is to use the knowledge of symptoms and the laboratory and imaging results from the given medical record and ask a question, combining the information and ask what the diagnosis is for this patient. Provide the 4 options: appendicitis, cholecystitis, diverticulitis, and pancreatitis. The "correct answer" field should have only the disease present in the medical record provided to you. This is named as the pathology, also found in the Discharge diagnosis or ICD Diagnosis field. Choose the option that is closest to the disease present in the medical record. Do not reveal the disease name when you phrase your question. Here is an example for you to learn from:
#         {
#         "26593491":
#            {
#             "Patient History":"__ year old otherwise healthy woman who presents with periumbilical -> RLQ pain. The patient was in her usual state of health until 10pm the night prior to presentation when she developed worsening periumbilical pain. She developed worsening nausea and NBNB vomiting. She presented to the ED for further evaluation. On ED presentation, she noted RLQ > periumbilical pain. She continued to have nausea but denied fevers, chills, diarrhea, sweats, recent weight loss, BRBPR, melena, chest pain, and SOB. Her last meal was the prior evening and her last drink of water was 5am the morning of presentation.  Past Medical History: None   Social History: ___ Family History: NC",    
#             "Physical Examination":"Exam on Admission Vitals: T 98.7 HR 76 BP 126/64 RR19 SpO2 100%RA GEN: A&O, lethargic but easily arousable, resting in stretcher HEENT: No scleral icterus, mucus membranes dry CV: RRR, No M/G/R PULM: Clear to auscultation b/l, No W/R/R ABD: Soft, nondistended. Tenderness to palpation in RLQ >periumbilical. No rebound or guarding. Negative ___ sign. No palpable masses. Ext: No ___ edema, ___ warm and well perfused.",
#             "Laboratory Tests": "(<FLUID>) <TEST>: <RESULT>(Blood) White Blood Cells: 17.1 K/uL(Blood) Red Blood Cells: 4.02 m/uL(Blood) Hemoglobin: 12.2 g/dL(Blood) Hematocrit: 36.9 %(Blood) MCV: 92.0 fL(Blood) MCH: 30.3 pg(Blood) MCHC: 33.1 g/dL(Blood) Platelet Count: 259.0 K/uL(Blood) Lymphocytes: 3.9 %(Blood) Absolute Lymphocyte Count: 0.66 K/uL(Blood) Basophils: 0.3 %(Blood) Absolute Basophil Count: 0.05 K/uL(Blood) Eosinophils: 0.0 %(Blood) Absolute Eosinophil Count: 0.0 K/uL(Blood) Monocytes: 5.0 %(Blood) Absolute Monocyte Count: 0.85 K/uL(Blood) Neutrophils: 90.2 %(Blood) Absolute Neutrophil Count: 15.41 K/uL(Blood) RDW: 12.6 %(Blood) RDW-SD: 41.9 fL(Blood) Alanine Aminotransferase (ALT): 24.0 IU/L(Blood) Asparate Aminotransferase (AST): 49.0 IU/L(Blood) Alkaline Phosphatase: 58.0 IU/L(Blood) Bilirubin, Total: 0.5 mg/dL(Blood) PT: 16.2 sec(Blood) INR(PT): UNABLE TO REPORT.(Blood) Albumin: 4.6 g/dL(Blood) Albumin: 4.6 g/dL(Blood) Urea Nitrogen: 18.0 mg/dL(Blood) Sodium: 138.0 mEq/L(Blood) Calcium, Total: 9.9 mg/dL(Blood) Chloride: 100.0 mEq/L(Blood) Creatinine: 0.8 mg/dL(Blood) Glucose: 135.0 mg/dL(Blood) Phosphate: 2.6 mg/dL(Blood) Potassium: 4.4 mEq/L(Urine) Urine Color: Yellow.(Urine) Urine Appearance: Hazy.(Urine) Urine Mucous: RARE.(Urine) Specific Gravity: 1.023  (Urine) Protein: 30.0 mg/dL(Urine) RBC: 4.0 #/hpf(Urine) WBC: 4.0 #/hpf(Urine) pH: 8.5 units(Urine) Bilirubin: NEG.(Urine) Glucose: NEG.(Urine) Urobilinogen: NEG.(Urine) Ketone: NEG.(Urine) Nitrite: NEG.(Urine) Leukocytes: SM .(Urine) Epithelial Cells: 1.0 #/hpf",
#             "Radiology": [
#                {"Note ID": "19955582-RR-23",
#                  "Modality": "CT",
#                  "Region": "Abdomen",
#                  "Exam Name": "CT ABD and PELVIS WITH CONTRAST",
#                  "Report": "EXAMINATION:CT ABDOMEN AND PELVIS WITH CONTRAST. :TECHNIQUE:Single phase split bolus contrast: MDCT axial images were acquiredthrough the abdomen and pelvis following intravenous contrast administrationwith split bolus technique.Coronal and sagittal reformations were performed and reviewed on PACS.DOSE:Total DLP (Body) = 452 mGy-cm.FINDINGS:LOWER CHEST:There is bibasilar atelectasis.  There is no pleural effusion.ABDOMEN:HEPATOBILIARY:The liver demonstrates homogenous attenuation throughout. Subcentimeter hypodensity in segment VII of the liver (series 2, image 29) istoo small to characterize.  There is mild periportal edema, likely from fluidresuscitation.  There is no evidence of intrahepatic or extrahepatic biliarydilatation.  The gallbladder is within normal limits.PANCREAS:The pancreas has normal attenuation throughout, without evidence offocal lesions or pancreatic ductal dilatation.  There is no peripancreaticstranding.SPLEEN:The spleen shows normal size and attenuation throughout, withoutevidence of focal lesions.ADRENALS:The right and left adrenal glands are normal in size and shape.URINARY:The kidneys are of normal and symmetric size with normal nephrogram. A subcentimeter hypodensity in the lower pole of the left kidney is too smallto characterize (series series 2, image 37)  There is no perinephricabnormality.GASTROINTESTINAL:The distal esophagus is normal without a hiatal hernia. Small bowel is normal in caliber without focal wall thickening.  Large bowelis also normal in caliber without focal wall thickening.  There is anappendicolith at the appendiceal base.  Distally the appendix is dilated up to13 mm with associated surrounding fat stranding.  There is air in the tip ofthe appendix, without evidence of extraluminal air.  There are nointra-abdominal fluid collections. PELVIS:The urinary bladder and distal ureters are unremarkable.  There is nofree fluid in the pelvis.REPRODUCTIVE ORGANS:The uterus and ovaries are unremarkable.LYMPH NODES:There is no retroperitoneal or mesenteric lymphadenopathy.  Thereis no pelvic or inguinal lymphadenopathy.VASCULAR:There is no abdominal aortic aneurysm. Mild atherosclerotic diseaseis noted.  There is dilation of the left gonadal vein.BONES:There is no evidence of worrisome osseous lesions or acute fracture. Limbus vertebra is seen at L4.  There is mild anterolisthesis of L4-L5.SOFT TISSUES:The abdominal and pelvic wall is within normal limits."
#                 },
#                 {"Note ID": "19955582-RR-24",
#                 "Modality": "CT",
#                 "Region": "Abdomen",
#                 "Exam Name": "CT ABD & PELVIS WITH CONTRAST",
#                 "Report": "TECHNIQUE:Single phase split bolus contrast: MDCT axial images were acquiredthrough the abdomen and pelvis following intravenous contrast administrationwith split bolus technique.Oral contrast was not administered.Coronal and sagittal reformations were performed and reviewed on PACS.DOSE:Acquisition sequence:   1) Spiral Acquisition 14.2 s, 48.7 cm; CTDIvol = 8.0 mGy (Body) DLP = 376.8mGy-cm. Total DLP (Body) = 390 mGy-cm.FINDINGS:LOWER CHEST:There are moderate bilateral nonhemorrhagic pleural effusionswhich are new from prior with associated atelectasis.ABDOMEN:HEPATOBILIARY:The liver demonstrates homogenous attenuation throughout. Multiple subcentimeter hypodensities in 
# the liver are too small tocharacterize by CT but appear unchanged from prior  There is no evidence ofintrahepatic or extrahepatic biliary dilatation.  The gallbladder is withinnormal limits.PANCREAS:The pancreas has normal attenuation throughout, without evidence offocal lesions or pancreatic ductal dilatation.  There is no peripancreaticstranding.SPLEEN:The spleen shows normal size and attenuation throughout, withoutevidence of focal lesions.ADRENALS:The right and left adrenal glands are normal in size and shape.URINARY:The 
# kidneys are of normal and symmetric size with normal nephrogram. There is a 3 mm hypodensity in the lower pole of the left kidney which is toosmall to characterize but statistically likely represents a cyst.  There is nohydronephrosis.  There is no perinephric abnormality.GASTROINTESTINAL:The stomach is unremarkable.  Small bowel loops demonstratenormal caliber, wall thickness, and enhancement throughout.  The colon andrectum are within normal limits.  The appendix is surgically absent.  There ishyperdense fluid extending from the right pericolic gutter adjacent to thesurgical site into the pelvis compatible with hemoperitoneum. Thoughevaluation is limited by single phase study, there is no obvious 
# area of focalcontrast extravasation to suggest active bleed.  Nonhemorrhagic free fluid isseen in the upper abdomen.PELVIS:The urinary bladder and distal ureters are unremarkable.REPRODUCTIVE ORGANS:The uterus is of normal size and enhancement. There is noevidence of adnexal abnormality bilaterally.LYMPH NODES:There is no retroperitoneal or mesenteric 
# lymphadenopathy.  Thereis no pelvic or inguinal lymphadenopathy.VASCULAR:There is no abdominal aortic aneurysm.  Mild atherosclerotic diseaseis noted.BONES:There is no evidence 
# of worrisome osseous lesions or acute fracture.SOFT TISSUES:Fluid is seen within the left anterior abdominal wallsubcutaneous tissues and extending between the abdominal musculature, likely post surgical.  Stranding is seen at the umbilicus likely postsurgical relatedto laparoscopic surgery.NOTIFICATION:The wet read was discussed by Dr. ___ with Dr. ___ on the___ ___ at 5:30 ___, 15 minutes after discovery of the findings."
#                 }
#                 ],
#                 "Discharge Diagnosis": "Appendicitis",
#                 "ICD Diagnosis": "[Acute appendicitis with localized peritonitis','Disseminated intravascular coagulation [defibrination syndrome]','Acute posthemorrhagic anemia','Postprocedural hemorrhage of skin and subcutaneous tissue following a dermatologic procedure','Postprocedural hemorrhage of a digestive system organ or structure following a 
# digestive system procedure','Coagulation defect, unspecified','Melanocytic nevi of trunk','Personal history of nicotine dependence','Removal of other organ (partial) (total) as 
# the cause of abnormal reaction of the patient, or of later complication, without mention of misadventure at the time of the procedure','Postprocedural fever','Unspecified place 
# in hospital as the place of occurrence of the external cause']",
#                 "Procedures Discharge": "['Laparoscopic appendectomy']",
#                 "Procedures ICD9": [],
#                 "Procedures ICD9 Title": [],
#                 "Procedures ICD10": "['0DTJ4ZZ', '0HB7XZX']",
#                 "Procedures ICD10 Title": "['Resection of Appendix, Percutaneous Endoscopic Approach','Excision of Abdomen Skin, External Approach, Diagnostic']"
#                 }
#             } 
#     {"id": 26593491, "question": "Provide the most likely final diagnosis of the following patient.@@@ PATIENT HISTORY @@@___ year old otherwise healthy woman who presents with periumbilical -> RLQ pain. The patient was in her usual state of health until 10pm the night prior to presentation when she developed worsening periumbilical pain. She developed worsening 
# nausea and NBNB vomiting. She presented to the ED for further evaluation. On ED presentation, she noted RLQ > periumbilical pain. She continued to have nausea but denied fevers, chills, diarrhea, sweats, recent weight loss, BRBPR, melena, chest pain, and SOB. Her last meal was the prior evening and her last drink of water was 5am the morning of presentation.   
#  Past Medical History: None   Social History: ___ Family History: NC
#  @@@ PHYSICAL EXAMINATION @@@Exam on Admission Vitals: T 98.7 HR 76 BP 126/64 RR19 SpO2 100%RA GEN: A&O, lethargic but easily arousable, resting in stretcher HEENT: No scleral icterus, mucus membranes dry CV: RRR, No M/G/R PULM: Clear to auscultation b/l, No W/R/R ABD: Soft, nondistended. Tenderness to palpation in RLQ >periumbilical. No rebound or guarding. Negative ___ sign. No palpable masses. Ext: No ___ edema, ___ warm and well perfused.  Exam
#  @@@ LABORATORY RESULTS @@@(<FLUID>) <TEST>: <RESULT>(Blood) White Blood Cells: 17.1 K/uL(Blood) Red Blood Cells: 4.02 m/uL(Blood) Hemoglobin: 12.2 g/dL(Blood) Hematocrit: 36.9 %(Blood) MCV: 92.0 fL(Blood) MCH: 30.3 pg(Blood) MCHC: 33.1 g/dL(Blood) Platelet Count: 259.0 K/uL(Blood) Lymphocytes: 3.9 %(Blood) Absolute Lymphocyte Count: 0.66 K/uL(Blood) Basophils: 0.3 %(Blood) Absolute Basophil Count: 0.05 K/uL(Blood) Eosinophils: 0.0 %
#  (Blood) Absolute Eosinophil Count: 0.0 K/uL(Blood) Monocytes: 5.0 %(Blood) Absolute Monocyte Count: 0.85 K/uL(Blood) Neutrophils: 90.2 %(Blood) Absolute Neutrophil Count: 15.41 K/uL(Blood) RDW: 12.6 %(Blood) RDW-SD: 41.9 fL(Blood) Alanine Aminotransferase (ALT): 24.0 IU/L(Blood) Asparate Aminotransferase (AST): 49.0 IU/L(Blood) Alkaline Phosphatase: 58.0 IU/L(Blood) Bilirubin, Total: 0.5 mg/dL(Blood) PT: 16.2 sec(Blood) INR(PT): UNABLE TO REPORT.
#  (Blood) Albumin: 4.6 g/dL(Blood) Albumin: 4.6 g/dL(Blood) Urea Nitrogen: 18.0 mg/dL(Blood) Sodium: 138.0 mEq/L(Blood) Calcium, Total: 9.9 mg/dL(Blood) Chloride: 100.0 mEq/L(Blood) Creatinine: 0.8 mg/dL(Blood) Glucose: 135.0 mg/dL(Blood) Phosphate: 2.6 mg/dL(Blood) Potassium: 4.4 mEq/L(Urine) Urine Color: Yellow.(Urine) Urine Appearance: Hazy.(Urine) Urine Mucous: RARE.(Urine) Specific Gravity: 1.023  (Urine) Protein: 30.0 mg/dL(Urine) RBC: 4.0 #/hpf(Urine) WBC: 4.0 #/hpf
#  (Urine) pH: 8.5 units(Urine) Bilirubin: NEG.(Urine) Glucose: NEG.(Urine) Urobilinogen: NEG.(Urine) Ketone: NEG.(Urine) Nitrite: NEG.(Urine) Leukocytes: SM .(Urine) Epithelial Cells: 1.0 #/hpf@@@ IMAGING RESULTS @@@CT AbdomenEXAMINATION:CT ABDOMEN AND PELVIS WITH CONTRAST.:TECHNIQUE:Single phase split bolus contrast: MDCT axial images were acquiredthrough the abdomen and pelvis following intravenous contrast administrationwith split bolus technique.
#  Coronal and sagittal reformations were performed and reviewed on PACS.DOSE:Total DLP (Body) = 452 mGy-cm.FINDINGS:LOWER CHEST:There is bibasilar atelectasis.  There is no pleural effusion.ABDOMEN:HEPATOBILIARY:The liver demonstrates homogenous attenuation throughout. Subcentimeter hypodensity in segment VII of the liver (series 2, image 29) istoo small to characterize.  There is mild periportal edema, likely from fluidresuscitation.  There is no evidence of intrahepatic or extrahepatic 
# biliarydilatation.  The gallbladder is within normal limits.PANCREAS:The pancreas has normal attenuation throughout, without evidence offocal lesions or pancreatic ductal dilatation.  There is no peripancreaticstranding.SPLEEN:The spleen shows normal size and attenuation throughout, withoutevidence of focal lesions.ADRENALS:The right and left adrenal glands are normal in size and shape.URINARY:The kidneys are of normal and symmetric size with normal nephrogram. A subcentimeter hypodensity in the lower pole of the left kidney 
# is too smallto characterize (series series 2, image 37)  There is no perinephricabnormality.GASTROINTESTINAL:The distal esophagus is normal without a hiatal hernia. Small bowel is normal in caliber without focal wall thickening.  Large bowelis also normal in caliber without focal wall thickening. There is anappendicolith at the appendiceal base.  Distally the appendix is dilated up to13 mm with associated surrounding fat stranding.  There is air in the tip ofthe appendix, without evidence of extraluminal air.  There are nointra-abdominal fluid collections.PELVIS:The urinary bladder and distal ureters are unremarkable.  There is nofree fluid in the pelvis.REPRODUCTIVE ORGANS:The uterus and ovaries are unremarkable.LYMPH NODES:There is no retroperitoneal or mesenteric lymphadenopathy.  Thereis no pelvic or inguinal lymphadenopathy.VASCULAR:There is no abdominal aortic aneurysm.  Mild atherosclerotic diseaseis noted.  There is dilation of the left gonadal vein.BONES:There is no evidence of worrisome osseous lesions or acute fracture. Limbus vertebra is seen at L4.  There is mild anterolisthesis of L4-L5.SOFT TISSUES:The abdominal and pelvic wall is within normal limits.CT AbdomenTECHNIQUE:Single phase split bolus contrast: MDCT axial images were acquiredthrough the abdomen and pelvis following intravenous contrast administrationwith split bolus technique.Oral contrast was not administered.Coronal and sagittal reformations were performed and reviewed on PACS.DOSE:Acquisition sequence:   1) Spiral Acquisition 14.2 s, 48.7 cm; CTDIvol = 8.0 mGy (Body) DLP = 376.8mGy-cm. Total DLP (Body) = 390 mGy-cm.FINDINGS:LOWER CHEST:There are moderate bilateral nonhemorrhagic pleural effusionswhich are new from prior with associated atelectasis.ABDOMEN:HEPATOBILIARY:The liver demonstrates homogenous attenuation throughout. Multiple subcentimeter hypodensities in the liver are too small to characterize by CT but appear unchanged from prior  There is no evidence ofintrahepatic or extrahepatic biliary dilatation.  The gallbladder is withinnormal limits.PANCREAS:The pancreas has normal attenuation throughout, 
# without evidence offocal lesions or pancreatic ductal dilatation.  There is no peripancreaticstranding.SPLEEN:The spleen shows normal size and attenuation throughout, withoutevidence of focal lesions.ADRENALS:The right and left adrenal glands are normal in size and shape.URINARY:The kidneys are of normal and symmetric size with normal nephrogram. There is a 3 mm hypodensity in the lower pole of the left kidney which is toosmall to characterize but statistically likely represents a cyst.  There is nohydronephrosis.  There is no perinephric abnormality.GASTROINTESTINAL:The stomach is unremarkable.  Small bowel loops demonstratenormal caliber, wall thickness, and enhancement throughout.  The colon andrectum are within normal limits.  The appendix is surgically absent.  There ishyperdense fluid extending from the right pericolic gutter adjacent to thesurgical site into the pelvis compatible with hemoperitoneum. Thoughevaluation is limited by single phase study, there is no obvious area of focalcontrast extravasation to suggest active bleed.  Nonhemorrhagic free fluid isseen in the upper abdomen.PELVIS:The urinary bladder and distal ureters are unremarkable.REPRODUCTIVE ORGANS:The uterus is of normal size and enhancement. There is noevidence of adnexal abnormality bilaterally.LYMPH NODES:There is no retroperitoneal or mesenteric lymphadenopathy.  There is no pelvic or inguinal lymphadenopathy.VASCULAR:There is no abdominal aortic aneurysm.  Mild atherosclerotic diseaseis noted.BONES:There is no evidence of worrisome osseous lesions or acute fracture.SOFT TISSUES:Fluid is 
# seen within the left anterior abdominal wallsubcutaneous tissues and extending between the abdominal musculature, likelypostsurgical.  Stranding is seen at the umbilicus likely 
# postsurgical relatedto laparoscopic surgery.NOTIFICATION:The wet read was discussed by Dr. ___ with Dr. ___ on the___ ___ at 5:30 ___, 15 minutes after discovery of the findings. What is the diagnosis for this patient?", 
# "dataset": "mimic-ext", 
# "answer_choices": {"A":"Appendicitis", "B":"Cholecystitis", "C":"Diverticulitis", "D":"Pancreatitis"}, 
# "correct_answer": "Appendicitis"}}
# - The JSON has to be in valid JSON format. Avoid writing ```json, , \, or extra quotes in between the json file elements. Just stick to the exact format shown in the above example.
#         - Keep shuffling the order in which these 4 answer choices are provided.
