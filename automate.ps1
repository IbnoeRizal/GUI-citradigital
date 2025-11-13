if(-not(get-command python)){
    write-output "intepreter python tidak ada, download terlebih dahulu";
    exit 0;
}
 
$isready = $true;
if(-not(Test-Path ".\.venv")){
    python -m venv .venv;
    $isready = $false;
}
if(-not $env:VIRTUAL_ENV){
    & .\.venv/Scripts/Activate;
}
if(-not $isready || -not(Test-Path ".venv/Lib/site-packages/cv2")){
    python -m pip install --upgrade pip;
    python -m pip install -r requirements.txt;
}
 
python .\gui.py;
