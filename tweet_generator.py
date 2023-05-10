import openai
import requests
import urllib.request
from bs4 import BeautifulSoup
import justext
import tiktoken
import re
import spacy
from spacy import displacy

NER = spacy.load("en_core_web_md")

openai.api_key = None

modelName = None
tweetCharacterLimit = None
chatGPTTokenLimit = None
safeContextLimit = None 
enc = None

tweetCharacterLimit = 280

searchAPI = None
cx = "46f8dd26da81f44df"

relevantEntityCategories = ["PERSON","NORP","ORG","WORK_OF_ART","LAW","EVENT","PRODUCT","NORP"] #https://www.kaggle.com/code/curiousprogrammer/entity-extraction-and-classification-using-spacy


def countTokens(text):
    tokenLst = enc.encode(text)
    tokenNum = len(tokenLst)
    return(tokenNum)

def search(query):
    responce = requests.get(f"https://www.googleapis.com/customsearch/v1?key={searchAPI}&cx={cx}&q={query}")
    return(responce)

def askGPT(msg, context, temperatureIn):
    completion = openai.ChatCompletion.create(model=modelName,temperature = temperatureIn,messages = [
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

def summariseOutput_GPT(initialText,gptRole,prompt,temperature):
    latestTweet = initialText
    while len(latestTweet)>tweetCharacterLimit:
        context = f"Imagine you are a {gptRole}. You have a Tweet that you want to post but it is a little long so you want to make it shorter"
        msg = f"Shorten the existing tweet a little bit while ensuring that it answers the query. The query is : '{prompt}'. The existing tweet is : '{initialText}'. The shortened tweet should preserve the vibe of the original tweet and be as engaging and clear."
        latestTweet = askGPT(msg, context,temperatureIn= temperature )
    
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

def selfImprovement(tweet,gptRole,prompt, context,temp,selfImprovementLimitCount):
    currTweet = tweet
    context = f"Imagine you are a {gptRole}. You have written a tweet '{currTweet}' that aims to answer the following query '{prompt}'. Now you want to improve that tweet. The original tweet was written based on the followin research : {context}"

    trace = [{"initial": currTweet}]
    for _ in range(1,selfImprovementLimitCount):
        msg = f"Please objectively evaluate the following tweet '{tweet}' with respect to it's clarity, engagement of the target audience, and it's ability to answer the following query/achieve following purpose '{prompt}'. Also suggest improvements without implementing them."
        feedback = askGPT(msg,context,temperatureIn = temp)

        msg = f"rewrite the tweet '{tweet}' based on the feedback '{feedback}' while only relying on the following research for factual statements : {context}"
        currTweet = askGPT(msg,context,temperatureIn = temp)

        trace.append({"feedback":feedback,"improved Tweet": currTweet})
        
    return currTweet,trace

def simplePromptTweetGeneration(prompt,numSources, gptRole, self_improvement,modelTemperature, namedEntitiesDisplayValue = False, preExtractedText = "",modelNameIn="gpt-3.5-turbo",currTweet="", requestType="gen",selfImprovementLimitCount = 1,apiKeyOpenAI = None,searchAPIKey = None  ):
    
    global modelName
    global chatGPTTokenLimit
    global safeContextLimit
    global enc
    global searchAPI

    print("loading...")
    
    openai.api_key = apiKeyOpenAI
    searchAPI = searchAPIKey

    modelName = modelNameIn

    if modelName == "gpt-4":
        chatGPTTokenLimit = 8192
    elif modelName == "gpt-3.5-turbo":
        chatGPTTokenLimit = 4097
    
    safeContextLimit = (chatGPTTokenLimit-tweetCharacterLimit)- 1000 # removed another 1000 to ensure that we have enough space for system message and query
    enc = tiktoken.encoding_for_model(modelName)

    # getting and processing context sources
    if preExtractedText == "":
        res = search(prompt)
        linkLst = getListOfLinks(res, numSources)
        text = extractTextFromResource(linkLst)
    else:
        linkLst = []
        text = preExtractedText

    if requestType == "gen":
        context = f"Imagine you are a {gptRole}"
        msg = f"Create a tweet that answers the following query '{prompt}' while making use of emoji if relevant. The response should be based on the following extracted text from a few different web pages: {text}"
    elif requestType == "imp":
        context = f"Imagine you are a {gptRole}. Imagine you wrote a tweet and you are not happy with it."
        msg = f"Improve the current tweet '{currTweet}' by doing the following: {prompt}. Here is some research that was used to generate the initial tweet: "+text

    
    output = askGPT(msg, context, temperatureIn=modelTemperature)

    # self improving tweet
    if self_improvement:
      output,trace = selfImprovement(output,gptRole,prompt,text, modelTemperature,selfImprovementLimitCount)
    else:
        trace = []

    # fixing up a tweet
    output = summariseOutput_GPT(output,gptRole,prompt,modelTemperature)
    output = removeHashtags(output)


    
    # named entities handling
    if namedEntitiesDisplayValue:
        namedEntities = extractRelevantNamedEntities(output, relevantEntityCategories)
    else:
        namedEntities = []

    return(output,namedEntities,linkLst,text,trace)

