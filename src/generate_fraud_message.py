import pandas as pd
import random

fraud_templates = [
    "Hi finance team please urgently wire ${amount} to the vendor account today",
    "This is the CEO. Process a transfer of ${amount} immediately",
    "Please transfer ${amount} to the updated vendor account",
    "I need an urgent payment of ${amount} sent today",
    "Kindly wire ${amount} immediately I am currently in a meeting",
    "Transfer ${amount} ASAP and confirm once completed",
    "Please update vendor bank details and send ${amount}",
    "Confidential request: wire ${amount} to the supplier today",
]

normal_templates = [
    "Reminder to review the quarterly financial report",
    "Please schedule the team meeting for tomorrow",
    "Invoice attached for last month's services",
    "Let's discuss the budget updates in today's meeting",
    "Please review the attached document",
    "Team lunch scheduled for Friday afternoon",
    "Reminder about the project planning meeting",
    "Please confirm receipt of this message",
]

def generate_dataset(size=10000):

    data = []

    for _ in range(size):

        if random.random() < 0.5:
            template = random.choice(fraud_templates)
            amount = random.randint(1000, 50000)
            message = template.replace("${amount}", str(amount))
            label = 1
        else:
            message = random.choice(normal_templates)
            label = 0

        data.append({
            "message": message,
            "label": label
        })

    return pd.DataFrame(data)


df = generate_dataset(10000)

df.to_csv("data/raw/fraud_messages.csv", index=False)

print("Dataset created with", len(df), "rows")
