from tkinter import *
from PIL import Image, ImageTk
from tkinter import messagebox
import mysql.connector as sql



def clear():
    userentry.delete(0,END)
    passentry.delete(0,END)
    conpassentry.delete(0,END)

def connect_db():
    if userentry.get()=='' or passentry.get()=='' or conpassentry.get()=='':
        messagebox.showerror('Error','All Fields Are Required')
    elif passentry.get() != conpassentry.get():
        messagebox.showerror('Error','Password Mismatch')
    else:
        try:
            con=sql.connect(host="localhost",user="root",password="Derick!@1221")
            mycursor=con.cursor()
        except:
            messagebox.showerror('Error','Database Connectivity Issue, Please Try Again')
            return
        try:
            query='create database farmerdb'
            mycursor.execute(query)
            query='Use farmerdb'
            mycursor.execute(query)
            query='create table farmerdata(id int auto_increment primary key not null,farmername varchar(40),password varchar(50))'
            mycursor.execute(query)
        except:
            mycursor.execute('use farmerdb')
        query='select * from farmerdata where farmername=%s'
        mycursor.execute(query,[(userentry.get())])
        row=mycursor.fetchone()
        if row != None:
            messagebox.showerror('Error','User Already Exist')
        else:
            query='insert into farmerdata(farmername,password) values(%s,%s)'
            mycursor.execute(query,(userentry.get(),passentry.get()))
            con.commit()
            con.close()
            messagebox.showinfo('Success','Registration is Successful')
            clear()
            signupwindow.destroy()
            import Login

def user_enter(event):
    if userentry.get() == 'Username':
        userentry.delete(0, END)

def pass_enter(event):
    if passentry.get() == 'Password':
        passentry.delete(0, END)

def pass_enter1(event):
    if conpassentry.get() == 'confirm Password':
        conpassentry.delete(0, END)

def login_page(event=None):  # Add event=None as a parameter
    signupwindow.destroy()  # Close the current window
    import Login

def change_cursor_to_hand(event):
    canvas.config(cursor="hand2")
def change_cursor_to_arrow(event):
    canvas.config(cursor="")

signupwindow = Tk()
signupwindow.geometry("874x494+50+50")
signupwindow.title("Signup page")

# Load and resize the image
image = Image.open("set2.png")
image = image.resize((874, 494))  # Resize the image to match the window size
bgimage = ImageTk.PhotoImage(image)

canvas = Canvas(signupwindow, width=874, height=494)
canvas.create_image(0, 0, image=bgimage, anchor=NW)
canvas.create_text(250,60, text="FARMER SIGNUP", fill="white", font=('times', 20, 'bold'))
canvas.create_text(140,120, text="User name :", fill="white", font=('times', 15, 'bold'))
canvas.create_text(135,180, text="Password :", fill="white", font=('times', 15, 'bold'))
canvas.create_text(170,240, text="Confirm Password :", fill="white", font=('times', 15, 'bold'))
canvas.create_text(240,370, text="-------------- OR --------------", fill="white", font=('times', 15, 'bold'))
canvas.create_text(185,400, text="Already have an Account?", fill="white", font=('times', 10, 'bold'))
loginbutton = canvas.create_text(285, 400, text="Login", fill="black", font=('times', 15, 'bold','underline'))
canvas.tag_bind(loginbutton, "<Button-1>", login_page)
canvas.tag_bind(loginbutton, "<Enter>", change_cursor_to_hand)# Bind the click event to the function
canvas.tag_bind(loginbutton, "<Leave>", change_cursor_to_arrow)
canvas.pack(fill="both", expand=True)


#a = Label(signupwindow, text="User name", font=("Microsoft yanei UI Light", "12"), fg='green', bg='#be9000', bd=0)
#a.place(x=110, y=150)

userentry = Entry(signupwindow, width=26, font=("Microsoft yanei UI Light", 11), fg='black',bg='white', bd=0)
userentry.place(x=110, y=140)
userentry.insert(0, 'Username')
userentry.bind('<FocusIn>', user_enter)

#b = Label(signupwindow, text="New Password", font=("Microsoft yanei UI Light", "12"), fg='green', bg='#be9000', bd=0)
#b.place(x=110, y=210)

passentry = Entry(signupwindow, width=26, font=("Microsoft yanei UI Light", 11), fg='black',bg='white', bd=0)
passentry.place(x=110, y=200)
passentry.insert(0, 'Password')
passentry.bind('<FocusIn>', pass_enter)

#c = Label(signupwindow, text="confirm Password", font=("Microsoft yanei UI Light", "12"), fg='green', bg='#be9000', bd=0)
#c.place(x=110, y=270)

conpassentry = Entry(signupwindow, width=26, font=("Microsoft yanei UI Light", 11), fg='black',bg='white', bd=0)
conpassentry.place(x=110, y=260)
conpassentry.insert(0, 'confirm Password')
conpassentry.bind('<FocusIn>', pass_enter1)

signupbutton = Button(signupwindow, width=20, height=2, text='SIGN UP', bd=0, bg='#05b3f7', cursor='hand2', font=("Microsoft yanei UI Light", 11), fg='black', activebackground='white',command=connect_db)
signupbutton.place(x=140, y=300)


signupwindow.mainloop()
