import openai
import requests
import urllib.request
from bs4 import BeautifulSoup
import justext
import tiktoken
import re
import spacy
from spacy import displacy
import en_core_web_trf

NER = en_core_web_trf.load()

openai.api_key = "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"
modelName = "gpt-3.5-turbo"

searchAPI = "AIzaSyDuqVqBQ6c3tFXjHvvgtazm_PhvbN-awUA"
cx = "46f8dd26da81f44df"

relevantEntityCategories = ["PERSON","NORP","ORG","WORK_OF_ART","LAW","EVENT","PRODUCT","NORP"] #https://www.kaggle.com/code/curiousprogrammer/entity-extraction-and-classification-using-spacy

tweetCharacterLimit = 280
chatGPTTokenLimit = 4097
safeContextLimit = chatGPTTokenLimit-tweetCharacterLimit

enc = tiktoken.encoding_for_model(modelName)

def countTokens(text):
    tokenLst = enc.encode(text)
    tokenNum = len(tokenLst)
    return(tokenNum)

def search(query):
    responce = requests.get(f"https://www.googleapis.com/customsearch/v1?key={searchAPI}&cx={cx}&q={query}")
    return(responce)

def askGPT(msg, context):
    completion = openai.ChatCompletion.create(model=modelName,temperature = 0.6,messages = [
                                {
                                    "role":"system","content":context,
                                    "role":"user","content":msg
                                 }])
    return(completion.choices[0].message.content)


def extractTextFromResource(linkLst):
    
    out = ""
    srsNum = 0
    stopFlag = False
    for link in linkLst:
        
        if stopFlag == True:
            break

        out = out + "\n" + "Web Source number " + str(srsNum) + "\n"
        response = requests.get(link)
        paragraphs = justext.justext(response.content, justext.get_stoplist("English"))
        
        for paragraph in paragraphs:
            tempOut = out + "\n" + paragraph.text
            
            if(countTokens(tempOut)>= safeContextLimit):
                stopFlag = True
                break

            if not paragraph.is_boilerplate:
                out = tempOut

        srsNum = srsNum +1
    
    return out

def summariseOutput_GPT(initialText,gptRole,prompt):
    latestTweet = initialText
    while len(latestTweet)>tweetCharacterLimit:
        context = f"Imagine you are a {gptRole}. You have a Tweet that you want to post but it is too long so you want to make it shorter"
        msg = f"Shorten the existing tweet a little bit while ensuring that it answers the query. The query is : '{prompt}'. The existing tweet is : {initialText}"
        latestTweet = askGPT(msg, context)
    
    return latestTweet


def getListOfLinks(searchResObj,bestK):
    linkLst = []
    for i in range(bestK):
        link = searchResObj.json()["items"][i]["link"] 
        linkLst.append(link)
    
    return(linkLst)

def removeHashtags(text):
    output_text = re.sub(r'#\w+', '', text)
    return(output_text)

def extractRelevantNamedEntities(text, spacyLabelList):
    doc = NER(text)

    relevantEntityTupples = [(w.text, w.label_) for w in doc.ents if w.label_ in spacyLabelList]
    relevantEntities = [ tup[0] for tup in relevantEntityTupples]
    
    return(relevantEntities)

def improvementLoop(gptRole,currTweet,text):
    while True:
        improvementRequest = input("Would you like any improvements done to the tweet? If not, press return : \n") or False

        if(improvementRequest == False):
            return(currTweet)
        
        context = f"Imagine you are a {gptRole}. Imagine you wrote a tweet and you are not happy with it."
        msg = f"Improve the current tweet '{currTweet}' by doing the following: {improvementRequest}"

        currTweet = askGPT(msg, context)

        print("Here is your improved Tweet : " + currTweet + "\n")


def simplePromptTweetGeneration(prompt,numSources, gptRole):
    # deals with prompts that are questions or "tell me about ..." type prompts
    res = search(prompt)
    linkLst = getListOfLinks(res, numSources)
    print("sources: " + str(linkLst) + "\n")
    text = extractTextFromResource(linkLst)

    context = f"Imagine you are a {gptRole}"
    msg = f"Create a tweet that answers the following query '{prompt}' while making use of emoji if relevant. The response should be based on the following extracted text from a few different web pages: {text}"
    
    output = askGPT(msg, context)
    print("initial tweet : " + output + "\n")

    output_better = summariseOutput_GPT(output,gptRole,prompt)
    output_better = removeHashtags(output_better)
    print("summarised tweet without hashtags : " + output_better + "\n")

    improvementLoop(gptRole,text)
    namedEntities = extractRelevantNamedEntities(output_better, relevantEntityCategories)

    print(namedEntities)

    return(output_better)

# prompt = input("Please provide prompt. Example: 'Are NFTs expensive?' : \n")
# defaultRole = 'A person building reputation as NFT expert by posting Tweets on Tweeter'
# role = input("What kind of person would you like to have written the tweet? The default is : '" + defaultRole + "'.Press return for default, else please enter below: \n") or defaultRole
# out = simplePromptTweetGeneration(prompt,5,role)

