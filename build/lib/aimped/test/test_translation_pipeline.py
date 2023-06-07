text = """Bei einer Knieluxation handelt es sich um eine besonders schwere Form der Multiligamentverletzung. Aufgrund der häufig hohen Verletzungskomplexität existiert eine Vielfalt unterschiedlicher Diagnostik- und Versorgungsstrategien. Mit dem Ziel evidenzbasierter Therapieempfehlungen richtet sich die S2e-Leitlinie „Knieluxation“ an alle an der Diagnostik und Therapie beteiligten Berufsgruppen (Orthopäden und Unfallchirurgen, Physiotherapeuten, ambulante/stationäre Operateure, Sportmediziner etc.) sowie Betroffene (Patient*innen mit Knieluxation) und Leistungserbringer (Krankenkassen, Rentenversicherungsträger). Diese umfasst neben der Darlegung konzeptioneller Unterschiede zwischen den Verletzungsentitäten die Besonderheiten der Diagnostik, konservativen und operativen Therapieoptionen auch Aspekte der Nachbehandlung vor dem Hintergrund des interdisziplinären Behandlungsansatzes einer schweren Knieverletzung.


            Bei einer Knieluxation handelt es sich um eine besonders schwere Form der Multiligamentverletzung. Aufgrund der häufig hohen Verletzungskomplexität existiert eine Vielfalt unterschiedlicher Diagnostik- und Versorgungsstrategien. Mit dem Ziel evidenzbasierter Therapieempfehlungen richtet sich die S2e-Leitlinie „Knieluxation“ an alle an der Diagnostik und Therapie beteiligten Berufsgruppen (Orthopäden und Unfallchirurgen, Physiotherapeuten, ambulante/stationäre Operateure, Sportmediziner etc.) sowie Betroffene (Patient*innen mit Knieluxation) und Leistungserbringer (Krankenkassen, Rentenversicherungsträger). Diese umfasst neben der Darlegung konzeptioneller Unterschiede zwischen den Verletzungsentitäten die Besonderheiten der Diagnostik, konservativen und operativen Therapieoptionen auch Aspekte der Nachbehandlung vor dem Hintergrund des interdisziplinären Behandlungsansatzes einer schweren Knieverletzung. 155 156 Eine SARS-CoV-2(„severe acute respiratory syndrome coronavirus 2“)-Infektion stellt für mehrere nephrologische Patientengruppen ein besonderes Risiko dar. Im Rahmen einer Pandemie sind dialysepflichtige Patienten ein besonders vulnerables Patientenkollektiv. Erste internationale Daten weisen auf eine deutlich erhöhte Mortalität von dialysepflichtigen Patienten im Rahmen einer SARS-CoV-2-Infektion hin. Aufgrund der erheblichen Unterschiede zwischen den Gesundheitssystemen, (staatlichen) Präventionsmaßnahmen, Therapiemöglichkeiten etc. in den verschiedenen Ländern weltweit sind lokale oder nationale Registerdaten im internationalen Kontext nicht ohne Weiteres übertragbar. Um zuverlässige Daten zur Prävalenz und Mortalität der SARS-CoV-2-Infektion bei dialysepflichtigen Patienten in Deutschland zu erheben, wurde unter dem Schirm der Deutschen Gesellschaft für Nephrologie ein Register für Dialysepatienten mit einer SARS-CoV-2-Infektion entwickelt. In diesem Rahmen erfolgt eine wöchentliche standardisierte Datenerhebung, die einen zeitnahen Überblick über die Fallzahlen und ggf. die Gewinnung neuer wissenschaftlicher Erkenntnisse ermöglichen soll. Insbesondere in der aktuellen Phase der Pandemie (Herbst 2020), die mit einer Vervielfachung der Anzahl der täglich Neuerkrankten innerhalb weniger Wochen einhergeht, sind solche Daten für eine Einschätzung und Anpassung des personellen, strukturellen und organisatorischen Bedarfs sowie der Kapazitäten für die Behandlung dieser Patienten hoch relevant.
"""

text2 = """Eine SARS-CoV-2(„severe acute respiratory syndrome coronavirus 2“)-Infektion stellt für mehrere nephrologische Patientengruppen ein besonderes Risiko dar. 
Im Rahmen einer Pandemie sind dialysepflichtige Patienten ein besonders vulnerables Patientenkollektiv. Erste internationale Daten weisen auf eine deutlich erhöhte Mortalität von dialysepflichtigen Patienten im Rahmen einer SARS-CoV-2-Infektion hin. Aufgrund der erheblichen Unterschiede zwischen den Gesundheitssystemen, (staatlichen) Präventionsmaßnahmen, Therapiemöglichkeiten etc. in den verschiedenen Ländern weltweit sind lokale oder nationale Registerdaten im internationalen Kontext nicht ohne Weiteres übertragbar. Um zuverlässige Daten zur Prävalenz und Mortalität der SARS-CoV-2-Infektion bei dialysepflichtigen Patienten in Deutschland zu erheben, wurde unter dem Schirm der Deutschen Gesellschaft für Nephrologie ein Register für Dialysepatienten mit einer SARS-CoV-2-Infektion entwickelt. 
In diesem Rahmen erfolgt eine wöchentliche standardisierte Datenerhebung, die einen zeitnahen Überblick über die Fallzahlen und ggf. die Gewinnung neuer wissenschaftlicher Erkenntnisse ermöglichen soll. Insbesondere in der aktuellen Phase der Pandemie (Herbst 2020), die mit einer Vervielfachung der Anzahl der täglich Neuerkrankten innerhalb weniger Wochen einhergeht, sind solche Daten für eine Einschätzung und Anpassung des personellen, strukturellen und organisatorischen Bedarfs sowie der Kapazitäten für die Behandlung dieser Patienten hoch relevant."""

from aimped.nlp.pipeline import Pipeline
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from aimped.nlp import translation


# Download and cache the model and tokenizer
checkpoint = "/media/forest/hdd/nlp-health-translation-base-de-en/nioyatech/nlp-health-translation-base-de-en"
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
device = 0
aimped = Pipeline(model=model, tokenizer=tokenizer, device=device)
aimped.translation_result([text1,text2],source_language="german", output_language="english")
