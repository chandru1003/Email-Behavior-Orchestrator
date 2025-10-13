
import os.path
import base64
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# If modifying these scopes, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly', 'https://www.googleapis.com/auth/gmail.send']

def get_gmail_service():
    """Shows basic usage of the Gmail API.
    Lists the user's Gmail labels.
    """
    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    try:
        # Call the Gmail API
        service = build('gmail', 'v1', credentials=creds)
        return service

    except HttpError as error:
        # TODO(developer) - Handle errors from gmail API.
        print(f'An error occurred: {error}')
        return None

def get_unread_emails(service):
    try:
        results = service.users().messages().list(userId='me', labelIds=['INBOX'], q="is:unread").execute()
        messages = results.get('messages', [])
        return messages
    except HttpError as error:
        print(f'An error occurred: {error}')
        return []

def get_email_content(service, msg_id):
    try:
        message = service.users().messages().get(userId='me', id=msg_id, format='full').execute()
        payload = message['payload']
        headers = payload['headers']
        subject = next(header['value'] for header in headers if header['name'] == 'Subject')
        sender = next(header['value'] for header in headers if header['name'] == 'From')

        if 'parts' in payload:
            parts = payload['parts']
            body = ""
            for part in parts:
                if part['mimeType'] == 'text/plain':
                    body = base64.urlsafe_b64decode(part['body']['data']).decode('utf-8')
                    break
        else:
            body = base64.urlsafe_b64decode(payload['body']['data']).decode('utf-8')

        return {'id': msg_id, 'subject': subject, 'from': sender, 'body': body}
    except HttpError as error:
        print(f'An error occurred: {error}')
        return None

def create_and_send_reply(service, to, subject, message_text, thread_id):
    try:
        message = (
            f"""From: me
To: {to}
Subject: Re: {subject}
In-Reply-To: {thread_id}
References: {thread_id}

{message_text}"""
        )
        encoded_message = base64.urlsafe_b64encode(message.encode('utf-8')).decode('utf-8')
        create_message = {'raw': encoded_message, 'threadId': thread_id}
        send_message = service.users().messages().send(userId='me', body=create_message).execute()
        print(f'Sent message to {to}, Message Id: {send_message["id"]}')
    except HttpError as error:
        print(f'An error occurred: {error}')

def apply_label(service, msg_id, label_name):
    """Applies a label to a specific email."""
    try:
        # First, find the ID of the label
        results = service.users().labels().list(userId='me').execute()
        labels = results.get('labels', [])
        label_id = next((l['id'] for l in labels if l['name'] == label_name), None)

        if not label_id:
            print(f"Label '{label_name}' not found.")
            return

        body = {'addLabelIds': [label_id]}
        service.users().messages().modify(userId='me', id=msg_id, body=body).execute()
        print(f"Applied label '{label_name}' to message {msg_id}")

    except HttpError as error:
        print(f'An error occurred while applying label: {error}')

def archive_email(service, msg_id):
    """Archives an email by removing the 'INBOX' label."""
    try:
        # To archive, we remove the 'INBOX' label from the message.
        body = {'removeLabelIds': ['INBOX']}
        service.users().messages().modify(userId='me', id=msg_id, body=body).execute()
        print(f"Archived message {msg_id}")
    except HttpError as error:
        print(f'An error occurred while archiving: {error}')
