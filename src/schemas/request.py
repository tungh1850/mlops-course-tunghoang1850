from enum import Enum

from pydantic import BaseModel


class occupation(str, Enum):
    EMPLYOEEE = "employee"
    WORKER = "worker"
    OTHER = "other"
    FREELANCER = "freelancer"
    STUDENT = "student"
    Business_Households = "business households"
    CASUAL_STAFF = "casual staff"
    RETIRED = "retired"
    FARMER = "farmer"
    MEDICAL_EGINEER = "medical engineer"
    ENGINEER = "engineer"
    HOUSE_WIFE = "house wife"
    MEDICAL_WORKER = "medical worker"
    LAWER_POLICE_ARMY_JOURNALIST = "lawyer, police, army, journalist"
    RETIREE = "retiree"


class gender(str, Enum):
    MALE = "male"
    FEMALE = "female"


class operating_system(str, Enum):
    IOS = "iOS"
    ANDROID = "Android"
    _0 = "0"


class phone_provider(str, Enum):
    VINA = "vinaphone"
    MOBI = "mobifone"
    VIETTEL = "viettel"


class PredictionRequest(BaseModel):
    Age: float
    occupation: occupation
    gender: gender
    operating_system: operating_system
    phone_provider: phone_provider
    ENQ_3M: float
    has_group2_debt_12m: float
    NUM_CC_NON_BANK: float
    NUM_NEW_LOAN_12M: float
    MID_TERM_COUNT_NON_BANK: float
    NUM_NEW_LOAN_6M: float
    OUTS_BAL_LOAN_M1: float
    LONG_TERM_AMOUNT: float
    ENQ_9M: float
    NUM_CC_BANK: float
    NUM_NEW_LOAN_3M: float
