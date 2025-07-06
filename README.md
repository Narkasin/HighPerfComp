# HighPerfComp
High Performance Computing Mod - Wolves Uni
Thomas Holloway - 1924424
6CS005 – High performance computing

Some things to note: To run on Windows
wsl --install 
(open in powershell and type this to install wsl)

sudo apt install GCC 
(you'll need this to build the files if building them manually)

--------------------------------------------------------------------------------------------------------
1.   Passwordcracker


So this program ran amazingly on a linux system. The only problem I had was trying to run it on Windows.
Even with using WSL and installing Ubuntu, it never seemed to run just right.

Make sure you’re in the directory of the files you want to build and enter
GCC -o endfile inputCfile.c -lcrypt

This spits out a file that can be used to run this with the crypt library (this was horrible to get working as intended) and uses the lcrypt library to include the crypt verification

Run the program like so:
open powershell or cmd (or type wsl in start) 
navigate to directory (or just type wsl in the file address bar)


./endfile (or output file name)


this should output the dialog options. 

I’ve included decrypt1 using passwordcracker.c

--------------------------------------------------------------------------------------------------------
2.    matrix multiplication


Very simple, this multiplies matrices in a start file and then spits the output into a results.txt file. 
IT uses the Matrixes.txt input filename
It asks to specify how many cpu threads you want this to work on (I did 10)


This outputs the results.txt file

--------------------------------------------------------------------------------------------------------
3.      password cracker in CUDA


CMD sometimes works if you have nvcc (nvidia cuda toolkit) installed
I used WSLto make this work properly 



Note: This program will test all possible combinations of 2 lowercase letters
followed by 2 digits (aa00 to zz99), which is 67,600 total combinations.
The GPU parallelisation should make this very fast. WOW!! SUPA FAST!!


--------------------------------------------------------------------------------------------------------
4.     Box Blur using CUDA


This is pretty simple, I've included the lodepng library too to ensure this compiles correctly 
This is using the nvcc with the previous question. LodePNG is used to enact edits on the PNG. 


