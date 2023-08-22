import openai
import csv
api_key = " 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"
openai.api_key = api_key

def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message["content"]

def process_output(response):
    output_lines = response.split("\n")
    feedbacks = []
    labels = []

    for line in output_lines:
        if line.strip() != "":
            parts = line.split(" - ")
            feedback = parts[0].strip()
            label = int(parts[1].strip())

            feedbacks.append(feedback)
            labels.append(label)

    return feedbacks, labels

def save_feedback_and_labels_to_csv(feedbacks, labels, output_file):
    if len(feedbacks) != len(labels):
        raise ValueError("The length of feedbacks and labels must be the same.")

    with open(output_file, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["feedback", "label"])
        for feedback, label in zip(feedbacks, labels):
            writer.writerow([feedback, label])

    print(f"Feedbacks and labels saved to {output_file}.")


doc = f"""
Expertiza provide the function for user to request an account, but the function is not perfect. It will enable Expertiza to handle the pending request for super-admin and add institution for user. 1. Currently, a new user can only choose institution from the dropdown. The new user should be able to add a new institution. 2. A place where a new user to write a brief introduction is needed in this page. 1. After super-admin or admin approves the request, make sure the record does not disappear on the page. And there should be an email send to email address offered by requester. 1. Modify the test of request new user method. 2. Add a new option for the drop-down bar of institution in the new user request page. 3. Add a new textarea in the new user request page, to input a new institution and introduction. 4. Add validation for new institution new institutions and introductions. 6. Add a new button on the layout. 8. Enable the system will send an email after the request is processed. So we need to add a new column for request_user table. The new table will be shown below. <image>. <image> <image> 1. Name: Instructor or Teaching Assistant request a new account 2. Actor: Instructor or Teaching Assistant 3. Other Participants: None 4. Precondition: None 5. Primary Sequence: 1. Click on ??煤Request account??霉. 3. Fill in a user name. 6. Select an institution. 8. Click on ??煤Request??霉. 2. Fill in the institution name. 1. Name: Admin or Super admin view list of new account requests 2. Actor: Admin or Super admin 3. Other Participants: None 4. Precondition: Instructor or Teaching Assistant has requested a new account 5. Primary Sequence: 1. Log in to Expertiza. 3. View the list of new account requests. 1. Alternative Flow: None 1. Name: Admin or Super admin accept a new account request 2. Actor: Admin or Super admin 3. Other Participants: None 4. Precondition: Instructor or Teaching Assistant has requested a new account 5. Primary Sequence: 1. Log in to Expertiza. 3. Select a request. 1. Alternative Flow: None 1. Name: Admin or Super admin reject a new account request 2. Actor: Admin or Super admin 3. Other Participants: None 4. Precondition: Instructor or Teaching Assistant has requested a new account 5. Primary Sequence: 1. Log in to Expertiza 2. Click on the link ??煤/users/list_pending_requested??霉. 3. Select a request. 1. Alternative Flow: None 1. Name: Admin or Super admin send an email to applicant 2. Actor: Admin or Super admin 3. Other Participants: None 4. Precondition: Instructor or Teaching Assistant has requested a new account 5. Primary Sequence: 1. Log in to Expertiza 2. Click on the link ??煤/users/list_pending_requested??霉. 3. Select a request. 4. Click on the email address. 1. Alternative Flow: None. <code> <code> <code> <code> <code>. <code> <code> <code> <code>. Below are the mock-up screens that explain ??煤new account acreation??霉 functionality in Expertiza. 1. 1.New user should click ??煤request account??霉 for requesting a new account, then redirect to the ??煤request new ??霉 page. <image> 1. 2.After redirect to ??煤request new ??霉 page, new users should enter their information, includes role, name, e-mail and institution. We can enter our new institution in this textbox. The email address is clickable. <image> 1. 5.If we click the email address, the email editor will pop out. <image>. In the first part, we test the request account feature. Finally, we test if the new user can signin with new account and password. 1.Now in the new user request page, the user can add a new institution by choose 'Other' and input their own institution. <image> 2.The new institution is able to saved into institutions table as a new record. <image> 5.The administrator can access the '/users/list_pending_requested' url via ??煤Administration > Show???>new requests??霉 menu <image> The admin can access the '/users/list_pending_requested' url via ??煤Administration > Manage>new requests??霉 menu <image> 6.??煤Email Address??霉 column in /users/list_pending_requested page clickable. <image> <image> 7.After super-admin or admin approves the request, the record does not disappear on the page. <image> 8.There is be an email send to email address offered by requester.And for super_admin and user. <image> <image>.
"""
feedback = f"""
Very good job of describing the changes made to the code.
It would have been helpful to have a more detailed description of what was done.
Also, it would have helped to have more of a description of the changes that were made.
The description of how the code is changed is not very clear.
There is no explanation of the code changes.

"""

prompt = f"""
You are a teacher responsible for grading homework.\
I will provide you with the student's documentation (doc) and the corresponding feedback on that documentation.\
Please evaluate the relevance of the feedback to the documentation. Note that the feedback consists of multiple sentences.\
The criterion for judging is whether the feedback is well combined with the content in the doc\
First of all, you should read and understand the content of the doc, and then read each sentence of the feedback to evaluate whether the feedback meets the criteria for interpretation
Use label to mark, the strong correlation is 1, and the weak correlation is 0.\
Please judge each sentence and return the original sentence and label\

"{doc}"
"{feedback}"

"""
response = get_completion(prompt)
print(response)

# 处理输出结果，提取feedback和correlation
feedbacks, labels = process_output(response)

# 指定CSV文件路径
output_csv_file = "E:\\desktop\\gears\\GPT-origin.csv"

# 调用函数，将反馈和标签写入CSV文件
save_feedback_and_labels_to_csv(feedbacks, labels, output_csv_file)