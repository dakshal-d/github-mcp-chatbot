import httpx

API_KEY="AIzaSyD8TSjUkwvX-WvDrIyrQWv0b4wmUwj-MqQ"

url="https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"

headers={
 "Authorization":f"Bearer {API_KEY}",
 "Content-Type":"application/json"
}

payload={
 "model":"gemini-2.0-flash",
 "messages":[{"role":"user","content":"hi"}]
}

r=httpx.post(url,headers=headers,json=payload)

print(r.status_code)
print(r.text)