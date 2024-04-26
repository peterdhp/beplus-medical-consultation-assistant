import streamlit as st
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
import io
import datetime
from audiorecorder import audiorecorder


from openai import OpenAI

if 'disabled' not in st.session_state:
    st.session_state.disabled = True

#st.session_state.temp_med_rec="[증상]\n[기타 특이사항]\n[진단]\n[치료, 처방 및 계획]"
with st.sidebar:
    st.image('logo.png', caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
    st.title("진료 도우미 (BETA ver.)")
    st.header("사용설명서")
    st.subheader("0.openAI API key 입력하기")
    st.markdown("https://m.blog.naver.com/mynameistk/223062993136 를 참고하여 APIkey를 아래에 입력한다.")
    openai_api_key = st.text_input('OpenAI API Key', type='password')
    if openai_api_key:
        st.session_state.disabled = False
    st.subheader("1.진료 준비하기")
    st.markdown("`진료기록 양식`에서 원하는 양식을 선택하거나 진료기록 텍스트 상자에 원하는 양식을 붙여넣기한다.")
    st.subheader("2.진료내용 녹음하기")
    st.markdown("`진료 녹음하기 🔴`을 눌러주고 진료를 진행한다. (진료가 시작되기 최소 3초전에는 녹음을 시작하는 것을 추천드립니다.)")
    st.subheader("3.진료 음성기록 변환하기")
    st.markdown("진료가 끝나면 `진료 녹음 끝내기 🟥`을 누르고 `진료 음성기록`이 완성되기를 기다린다. (잘못 기록한 의학 용어 등이 있을 경우 바로 수정 가능)")
    st.subheader("4.진료기록 자동 완성하기")
    st.markdown("`✍🏻 진료기록 자동 완성 및 ✅ 진료 내용 검토`을 눌러 진료기록이 완성되고 검토되기를 기다린다.")
    st.subheader("5.새로고침")
    st.markdown("`새로운 환자`을 눌러 이전 진료기록을 지운다.")

if "total_cost" not in st.session_state:
    st.session_state.totalcost = 0

transcript = """[환자] : 안녕하세요
[의사] : 네 안녕하세요 환자분, 어깨가 아프셔서 방문하셨군요. 운동범위를 확인해봐야될 것 같아요. 괜찮으시겠어요?
[환자] : 네, 그럼요
[의사] : 왼팔을 최대한 한번 들어보시겠어요?
[환자] : 네
[의사] : 왼팔은 170도 정도 되시네요. 제가 좀 더 올려볼게요.
[환자] : 아아 아파요
[의사] : 왼 어깨도 좋진 않으시네요. 평소에 안 불편하셨어요?
[환자] : 병원 올 정도는 아니어서요.
[의사] : 자 이번엔 오른팔 올려볼게요.
[환자] : 아아 여기가 최대예요.
[의사] : 120정도 밖에 안되시네요. 고생하셨겠어요.
[환자] : 네 아무것도 못하고 있죠 뭐
[의사] : 제가 조금 더 올려볼게요.
[환자] : 아아아아아아 아파요.
[의사] : 네 다 되셨어요. 오른쪽 어깨는 회전근개 파열이 의심되시는 상황이시고 왼쪽 어깨는 오십견이 오신거 같네요.
[환자] : 선생님 어떻게 빨리 낫거나 할 수 있는 방법이 없나요? 파열 됐으면 수술 같은 것을 받아야하나요?
[의사] : 수술은 파열이 어느정도 되었는지 살펴보고 말씀드릴 수 있어요. 검사를 진행해봐야될 것 같습니다. 예약을 잡아드리겠습니다.
[환자] : 무슨 검사인가요?
[의사] : MRI 검사라고 통속에 들어가서 사진을 찍는겁니다. 왼쪽도 찍는김에 같이 찍을게요. 그리고 그전에 통증 조절을 위해 진통제 처방 드리겠습니다.
[환자] : 아이고 감사합니다. 무통주사 이런거는 없을까요? 너무 힘듭니다.
[의사] : 놔드릴 수는 있는 비용이 좀 발생하세요.
[환자] : 실비 보험이 들어있어서 괜찮습니다. 놔주세요.
[의사] : 네 그럼 무통주사도 같이 놔드릴게요. 고혈압이나 당뇨병 같은 기저질환은 없으세요?
[환자] : 네 다른건 다 괜찮고 건강합니다.
[의사] : 네 알겠습니다. 밖에서 기다리시면 처방전이랑 검사 예약 잡아드릴게요.
[환자] : 혹시 보험사에 제출할 세부진료내역서도 
[의사] : 네 해드릴게요. 기다리세요
"""
transcript = """안녕하세요. 네 안녕하세요 환자분, 어깨가 아프셔서 방문하셨군요. 운동범위를 확인해봐야될 것 같아요. 괜찮으시겠어요? 네, 그럼요. 왼팔을 최대한 한번 들어보시겠어요? 네 왼팔은 170도 정도 되시네요. 제가 좀 더 올려볼게요. 아아 아파요. 왼 어깨도 좋진 않으시네요. 평소에 안 불편하셨어요? 병원 올 정도는 아니어서요. 자 이번엔 오른팔 올려볼게요. 아아 여기가 최대예요. 120정도 밖에 안되시네요. 고생하셨겠어요. 네 아무것도 못하고 있죠 뭐. 제가 조금 더 올려볼게요. 아아아아아아 아파요. 네 다 되셨어요. 오른쪽 어깨는 회전근개 파열이 의심되시는 상황이시고 왼쪽 어깨는 오십견이 오신거 같네요. 선생님 어떻게 빨리 낫거나 할 수 있는 방법이 없나요? 파열 됐으면 수술 같은 것을 받아야하나요? 수술은 파열이 어느정도 되었는지 살펴보고 말씀드릴 수 있어요. 검사를 진행해봐야될 것 같습니다. 예약을 잡아드리겠습니다. 무슨 검사인가요? MRI 검사라고 통속에 들어가서 사진을 찍는겁니다. 왼쪽도 찍는김에 같이 찍을게요. 그리고 그전에 통증 조절을 위해 진통제 처방 드리겠습니다. 아이고 감사합니다. 무통주사 이런거는 없을까요? 너무 힘듭니다. 놔드릴 수는 있는 비용이 좀 발생하세요. 실비 보험이 들어있어서 괜찮습니다. 놔주세요. 네 그럼 무통주사도 같이 놔드릴게요. 고혈압이나 당뇨병 같은 기저질환은 없으세요? 네 다른건 다 괜찮고 건강합니다. 네 알겠습니다. 밖에서 기다리시면 처방전이랑 검사 예약 잡아드릴게요. 혹시 보험사에 제출할 세부진료내역서도. 네 해드릴게요. 기다리세요"""
def refresh():
    st.session_state.totalcost = 0
    st.session_state.format_type = '없음'
    st.session_state.transcript =''
    st.session_state.temp_medical_record ="[현병력]\n\n[ROS]\n\n[신체검진]\n\n[impression]"

def medical_record(transcript):
    """문진 내용을 기반으로 질문을 함"""
    
    prompt_template = """Given the transcript, write a semi-filled medical report of the patient. Only fill in the form based on the transcript. 
                
[transcript]
{transcript}

The output(except the diagnosis) should be in Korean. Here is an example :

[병력] #최대한 구체적으로 빠짐없이, 왜 내원하게 되었는지 설명
상기 환자는 67세 남자로, 3개월 전부터 복부 통증을 호소하였음. 1주일 전부터 증상이 심해지다가 구토와 어지러움을 느껴 내원함
    
[증상] #양성 증상에서 참고하기 좋은 구체적인 내용이 있을 경우 같이 표시해줘
구토(+, 하루 4번)
복통(+, NRS 5)
악화요인(+, 식사하고 나서 심해짐)
토혈(-)
    
[기타 특이사항]
V/S : 130/90, 88, 36.5도
약물력 : 특이사항 없음
가족력 : 모름
    
[진단] #진단명은 영어로해줘. 예상 되는 진단 5개를 알려주고 왜 그렇게 생각했는지와 해야할 검사들을 알려줘. 진단을 할 때 특별히 유의할 점도 정리해줘
R/O peptic ulcer(복부 통증이 있고, 어지러움을 느낌, 확인을 위해 위내시경을 시행)
DDx1. reflux esophagitis(식사를 하고 나서 악화됨, 위내시경 시행)
DDx2. gastric cancer (3개월전부터 호소, 위내시경 시행)
DDx3. functional dyspepsia (3개월전부터 호소, 경과관찰)
DDx4. trauma (복부 통증, xray로 골절 확인)

주의할 점 : 67세 남성으로 위암을 배제할 수 없으므로 건강검진 시행여부 확인
    
[치료,처방 및 계획]
CBC 시행
위내시경 시행

-----
                
                """

    prompt = PromptTemplate.from_template(prompt_template)
    llm = ChatOpenAI(model_name="gpt-4-turbo", temperature = 0)
    output_parser = StrOutputParser()

    chain = prompt | llm | output_parser    
    
    output = chain.invoke({"transcript" : transcript})
    return output

def medical_record_voicecomplete():
    """문진 내용을 기반으로 질문을 함"""
    
    
    
    prompt_template = """Given a transcript of a patient consultation and a incomplete medical record, complete and edit the medical record. 
Only complete or edit the medical record based on the information given. For the physical examination KEEP THE FORMAT and only change what is necessary, also explain in Korean when changed.

[transcript]
{transcript}

[incomplete medical record]
{incomplete_medrec}

-----"""
    
    #user_type_mapping = {'human': '[patient]', 'ai': '[doctor]'}
    #msg_log_text = "\n".join(f"{user_type_mapping[sender]} : {message}" for sender, message in transcript)
    
    prompt = PromptTemplate.from_template(prompt_template)
    
    llm = ChatOpenAI(model_name="gpt-4-turbo", temperature = 0)
    output_parser = StrOutputParser()

    chain = prompt | llm | output_parser    
    
    return chain

def update_text():
    if st.session_state.format_type == '없음' and st.session_state.temp_medical_record == "":
        with st.spinner('음성 녹음을 바탕으로 진료 기록을 완성하고 있습니다...'):
            st.session_state.LLM_medrecord = medical_record(transcript=st.session_state.transcript)
    else :    
        chain = medical_record_voicecomplete()
        with st.spinner('음성 녹음을 바탕으로 진료 기록을 완성하고 있습니다...'):
            st.session_state.LLM_medrecord = chain.invoke({"transcript" : st.session_state.transcript, "incomplete_medrec" : st.session_state.temp_medical_record})
    st.session_state.temp_medical_record = st.session_state.LLM_medrecord 
   
def update_text_advise():
    if st.session_state.format_type == '없음' and st.session_state.temp_medical_record == "":
        with st.spinner('음성 녹음을 바탕으로 진료 기록을 완성하고 있습니다...'):
            st.session_state.LLM_medrecord = medical_record(transcript=st.session_state.transcript)
    else :    
        chain = medical_record_voicecomplete()
        with st.spinner('음성 녹음을 바탕으로 진료 기록을 완성하고 있습니다...'):
            st.session_state.LLM_medrecord = chain.invoke({"transcript" : st.session_state.transcript, "incomplete_medrec" : st.session_state.temp_medical_record})
    st.session_state.temp_medical_record = st.session_state.LLM_medrecord 
    st.success("진료 기록을 성공적으로 완성하였습니다.")
    with st.spinner('진료 내용을 검토하고 있습니다...'):
        output = medical_advisor(st.session_state.temp_medical_record,st.session_state.transcript)
    st.session_state.temp_medical_record += '\n\n'+ output
    st.success("진료 내용 검토 성공적으로 완료 되었습니다.")
 
def recorddemo():
    st.session_state.transcript = """안녕하세요. 네 안녕하세요 환자분, 어깨가 아프셔서 방문하셨군요. 운동범위를 확인해봐야될 것 같아요. 괜찮으시겠어요? 네, 그럼요. 왼팔을 최대한 한번 들어보시겠어요? 네 왼팔은 150도 정도 되시네요. 제가 좀 더 올려볼게요. 아아 아파요. 왼 어깨도 좋진 않으시네요. 평소에 안 불편하셨어요? 병원 올 정도는 아니어서요. 자 이번엔 오른팔 올려볼게요. 아아 여기가 최대예요. 120정도 밖에 안되시네요. 고생하셨겠어요. 네 아무것도 못하고 있죠 뭐. 제가 조금 더 올려볼게요. 아아아아아아 아파요. 네 다 되셨어요. 오른쪽 어깨는 회전근개 파열이 의심되시는 상황이시고 왼쪽 어깨는 오십견이 오신거 같네요. 선생님 어떻게 빨리 낫거나 할 수 있는 방법이 없나요? 파열 됐으면 수술 같은 것을 받아야하나요? 수술은 파열이 어느정도 되었는지 살펴보고 말씀드릴 수 있어요. 검사를 진행해봐야될 것 같습니다. 예약을 잡아드리겠습니다. 무슨 검사인가요? MRI 검사라고 통속에 들어가서 사진을 찍는겁니다. 왼쪽도 찍는김에 같이 찍을게요. 그리고 그전에 통증 조절을 위해 진통제 처방 드리겠습니다. 아이고 감사합니다. 무통주사 이런거는 없을까요? 너무 힘듭니다. 놔드릴 수는 있는 비용이 좀 발생하세요. 실비 보험이 들어있어서 괜찮습니다. 놔주세요. 네 그럼 무통주사도 같이 놔드릴게요. 고혈압이나 당뇨병 같은 기저질환은 없으세요? 네 다른건 다 괜찮고 건강합니다. 네 알겠습니다. 밖에서 기다리시면 처방전이랑 검사 예약 잡아드릴게요. 혹시 보험사에 제출할 세부진료내역서도. 네 해드릴게요. 기다리세요"""

def completedemo():
    st.session_state.temp_medical_record = """[현병력]
환자는 양쪽 어깨 통증을 호소하며 내원하였습니다. 왼쪽 어깨는 오십견이 의심되며, 오른쪽 어깨는 회전근개 파열이 의심됩니다.

[ROS]
어깨 통증 외에 다른 증상은 보고되지 않았습니다.

[신체검진]
<shoulder ROM>
Lt. abduction/adduction = 150/30 (왼쪽 어깨의 abduction이 150도로 확인되었습니다. 이는 환자가 왼쪽 팔을 최대로 들었을 때의 각도입니다.)
Rt. abduction/adduction = 120/30 (오른쪽 어깨의 abduction이 120도로 확인되었습니다. 환자가 오른쪽 팔을 최대로 들었을 때의 각도로, 통증으로 인해 제한된 범위를 보였습니다.)
Lt. extension/flexion = 50/150
Rt. extension/flexion = 50/150

[impression]
왼쪽 어깨: 오십견 의심
오른쪽 어깨: 회전근개 파열 의심

[추가 정보]
환자는 고혈압이나 당뇨병과 같은 기저질환은 없으며, 일반적으로 건강한 상태입니다. MRI 검사 예약 및 진통제 처방이 진행될 예정입니다. 또한, 환자의 요청에 따라 통증 조절을 위한 무통주사도 처방될 예정입니다. 환자는 실비 보험에 가입되어 있어 비용에 대한 부담이 적습니다."""

def format_retriever(format_type):
    
    format_lib ={}
    
    format_lib["없음"] = ""
    format_lib["기본"] = "[현병력]\n\n[ROS]\n\n[신체검진]\n\n[impression]"
    format_lib["어깨통증"] = """[현병력]
    
[ROS]
    
[신체검진]
<shoulder ROM>
Lt. abduction/adduction = 150/30
Rt. abduction/adduction = 150/30
Lt. extension/flexion = 50/150
Rt. extension/flexion = 50/150

[impression]"""
    
    output = format_lib.get(format_type)
    
    return output

def call_format():
    st.session_state.temp_medical_record = format_retriever(st.session_state.format_type)

def advise(): 
    output = medical_advisor(st.session_state.temp_medical_record,st.session_state.transcript)
    st.session_state.temp_medical_record += '\n\n'+ output
    st.success("진료 내용 검토 성공적으로 완료 되었습니다.")

def medical_advisor(medical_record, transcript):
    prompt_template = """Given a transcript of a patient consultation and a complete medical record, give medical advice in Korean.
For example, 
1. Check drug contraindication
2. Check test contraindication
2. Write list of things that were explained to the patient
3. Check if all important information was explained to the patient

[transcript]
{transcript}

[complete medical record]
{medical_record}
-----"""

    prompt = PromptTemplate.from_template(prompt_template)
    
    llm = ChatOpenAI(model_name="gpt-4-turbo", temperature = 0)
    output_parser = StrOutputParser()

    chain = prompt | llm | output_parser
    output = chain.invoke({"transcript" : transcript, "medical_record" : medical_record})
    
    return output

class NamedBytesIO(io.BytesIO):
    def __init__(self, buffer=None, name=None):
        super().__init__(buffer)
        self.name = name

st.selectbox("진료기록 양식", options=['없음', '기본', '어깨통증'],index=1,on_change=call_format, key='format_type')

st.text_area('진료 기록', value="[현병력]\n\n[ROS]\n\n[신체검진]\n\n[Impression]", height=600, key='temp_medical_record')

#timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

#byte_io = io.BytesIO()
#audio.export(byte_io, format='mp3')
#byte_io.seek(0)


if not openai_api_key.startswith('sk-'):
    st.warning('Please enter your OpenAI API key!', icon='⚠')
if openai_api_key.startswith('sk-'):
    client = OpenAI()
    audio = audiorecorder(start_prompt="진료 녹음하기 🔴", stop_prompt="진료 녹음 끝내기 🟥", pause_prompt="", key=None)

if openai_api_key.startswith('sk-') and len(audio)>0.1:
    with st.spinner('음성 녹음을 받아적고 있습니다...'):
        asr_result = client.audio.transcriptions.create(model="whisper-1", language= "ko",file= NamedBytesIO(audio.export().read(), name="audio.wav"))
    st.session_state.transcript += '\n'+ asr_result.text       
st.text_area("진료 음성기록", key='transcript')
st.button('✍🏻 진료기록 자동 완성 및 ✅ 진료 내용 검토',on_click=update_text_advise,disabled= st.session_state.disabled)
st.button('🔄 새로운 환자',on_click=refresh,key='refreshbutton')
   
#encoded_image = base64.b64encode(open("logo.png", "rb").read()).decode()



    
    #st.button("음성녹음 Demo",on_click=recorddemo)
    #st.button("자동작성완료 Demo",on_click=completedemo)
    #st.session_state
