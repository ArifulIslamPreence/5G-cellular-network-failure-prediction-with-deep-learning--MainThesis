@echo off
setlocal ENABLEDELAYEDEXPANSION

SET BFN = ../Data_preprocessing/prediction/codebase/Large_Dataset/new_combined.csv

REM number of rows per file
SET LPF=10000000

SET SFN= new_combined 

SET SFX=%BFN:~-3%
SET /A LineNum=0
SET /A FileNum=1
For /F "delims==" %%l in (%BFN%) Do (
    SET /A LineNum+=1
    echo %%l >> %SFN%!FileNum!.%SFX%
    if !LineNum! EQU !LPF! (
        SET /A LineNum=0
        SET /A FileNum+=1
    )
)
endlocal
Pause