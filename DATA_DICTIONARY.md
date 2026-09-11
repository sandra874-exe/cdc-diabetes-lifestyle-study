# Data Dictionary

The project uses the CDC BRFSS 2015 diabetes-health-indicators variables.

| Variable | Meaning used in the project | Coding / scale |
|---|---|---|
| `Diabetes_012` | Three-class diabetes status | 0 = no diabetes, 1 = prediabetes, 2 = diabetes |
| `Diabetes_binary` | Binary outcome for secondary analyses | 0 = no diabetes, 1 = prediabetes or diabetes |
| `HighBP` | High blood pressure indicator | 0/1 |
| `HighChol` | High cholesterol indicator | 0/1 |
| `CholCheck` | Cholesterol check indicator | 0/1 |
| `BMI` | Body Mass Index | Continuous |
| `Smoker` | Smoking indicator | 0/1 |
| `Stroke` | Stroke history indicator | 0/1 |
| `HeartDiseaseorAttack` | Coronary heart disease/heart attack indicator | 0/1 |
| `PhysActivity` | Physical activity indicator | 0/1 |
| `Fruits` | Fruit consumption indicator | 0/1 |
| `Veggies` | Vegetable consumption indicator | 0/1 |
| `HvyAlcoholConsump` | Heavy alcohol consumption indicator | 0/1 |
| `AnyHealthcare` | Healthcare coverage indicator | 0/1 |
| `NoDocbcCost` | Cost prevented a doctor visit indicator | 0/1 |
| `GenHlth` | Self-reported general health | 1 = excellent ... 5 = poor |
| `MentHlth` | Number of days in the past 30 days when mental health was not good | 0–30 |
| `PhysHlth` | Number of days in the past 30 days when physical health was not good | 0–30 |
| `DiffWalk` | Difficulty walking/climbing stairs indicator | 0/1 |
| `Sex` | Sex indicator | 0/1 |
| `Age` | BRFSS age category | 1–13 |
| `Education` | Education category | 1–6 |
| `Income` | Income category | 1–8 |
| `Is_Obese` | Project-derived obesity indicator from BMI | 1 if BMI >= 30 |
| `Metabolic_Factor_Count` | Project-defined count of HighBP + HighChol + Is_Obese | 0–3 |
| `Non_Smoker` | Project-derived inverse of `Smoker` | 0/1 |
| `Healthy_Habits` | Project-defined count of physical activity + fruit + vegetables + non-smoking | 0–4 |
| `Combined_Health_Burden` | Project-defined `MentHlth + PhysHlth` | 0–60 theoretical sum; not unique days |

### Important interpretation notes

`Age`, `Education`, and `Income` are coded categories rather than raw years or dollar amounts.

`Healthy_Habits`, `Metabolic_Factor_Count`, and `Combined_Health_Burden` are project-defined analytical features and are not validated clinical measures.

`Diabetes_binary = 1` must always be described as **prediabetes/diabetes**.
