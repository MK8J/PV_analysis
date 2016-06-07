
import sys
import os
import numpy as np
import matplotlib.pylab as plt
if sys.platform == "win32":
    from win32com.client import Dispatch
elif:
    import openpyxls as pyxl

## To do:
    #1. move away from  dispatch and uses openpyxls

def SintonPre2014_ExtractRawData(File, use_wkdkr=None, Plot=False):
    '''
    This grabs the calculated data from sinton
    Extracts columns A-G of the raw data page. Can't extract H-I as it is
    not complete data and this would take some more work
    This outputs the data as a structured array, with the names as the column
    '''
    # import the Dispatch library, get a reference to an Excel instance
    xlApp = Dispatch("Excel.Application")
    # Set the Excel file name, working directory and path
    # This is encase you don't want to use the directory you are in
    if use_wkdkr is None:
        use_wkdkr = os.path.dirname(__file__)
        print 'defult file path used:', use_wkdkr

    File_path = os.path.join(use_wkdkr, File)

    # opens the excel book
    xlBook = xlApp.Workbooks.Open(File_path)

    # make Excel visible (1 is True, 0 is False)
    xlApp.Visible = 0

    # makes a reference to the RawData page
    xlSheet = xlBook.Sheets('Calc')

    # Get the values from the page
    Values1 = np.asarray(xlSheet.Range("A15:I130").Value).T
    Values2 = np.asarray(xlSheet.Range("K15:P130").Value).T
    headers = tuple([[j.encode('utf8') for j in i]
                     for i in xlSheet.Range('A8:U8').Value][0])

    xlBook.Close(SaveChanges=0)
    xlApp.Application.Quit()

    Values = np.vstack((Values1, Values2))

    # this named array is not working
    # Out = Values.T.view(dtype=zip(headers, ['float64'] * len(headers))).copy()

    # Too see it working
    # print Out.dtype.names
    if Plot:
        plt.plot(Out['Apparent Carrier Density'], Out['TAU'], '--')
        plt.loglog()

    return Values


def Sinton2014_Settings(File):
    # This grabs the calibration constants for the coil and reference
    # This outputs them as a dictionary

    xlApp = Dispatch("Excel.Application")
    # Set the Excel file name, working directory and path
    # This is encase you don't want to use the directory you are in
    working_dir = r"C:/Users/z3186867/Desktop/"

    File_path = os.path.join(working_dir, File)
    xlBook = xlApp.Workbooks.Open(File_path)

    # make Excel visible (1 is True, 0 is False)
    xlApp.Visible = 0

    # makes a reference to the RawData page
    xlSheet = xlBook.Sheets('Settings')

    Ref = float(xlSheet.Range('C5').Value)
    A = float(xlSheet.Range('C6').Value)
    B = float(xlSheet.Range('C7').Value)
    C = float(xlSheet.Range('C8').Value)
    Air = float(xlSheet.Range('C10').Value)

    xlBook.Close(SaveChanges=0)
    xlApp.Application.Quit()
    Dic = locals()
    for i in ['xlApp', 'xlBook', 'working_dir', 'xlSheet', 'File_path']:
        del Dic[i]
    # print Dic
    return Dic


def Sinton2014_set_UserData(File, settings_dic, Dir=None):
    # This grabs the user entered data about the sample
    # This outputs them as a dictionary
    xlApp = Dispatch("Excel.Application")
    # Set the Excel file name, working directory and path
    # This is encase you don't want to use the directory you are in
    if Dir is None:
        Dir = r"C:\Users\z3186867\Desktop\\"

    File_path = os.path.join(Dir, File)
    # print File_path
    # xlApp.Visible = True
    xlBook = xlApp.Workbooks.Open(File_path)

    # make Excel visible (1 is True, 0 is False)
    xlApp.Visible = 0

    # makes a reference to the RawData page
    xlSheet = xlBook.Sheets('User')

    # Grabbing the data and assigning it a nae
    xlSheet.Range('A6').Value = settings_dic['WaferName']
    xlSheet.Range('B6').Value = settings_dic['Thickness']
    print settings_dic['Resisitivity'], settings_dic
    xlSheet.Range('C6').Value = settings_dic['Resisitivity']
    xlSheet.Range('D6').Value = settings_dic['SampleType']
    xlSheet.Range('H6').Value = settings_dic['AnalysisMode']
    xlSheet.Range('E6').Value = settings_dic['OpticalConstant']

    xlBook.Close(SaveChanges=1)
    xlApp.Application.Quit()
    pass


def Sinton2014_ExtractUserData(File):
    # This grabs the user entered data about the sample
    # This outputs them as a dictionary
    xlApp = Dispatch("Excel.Application")
    # Set the Excel file name, working directory and path
    # This is encase you don't want to use the directory you are in

    File_path = os.path.join(File)
    # print File_path
    # xlApp.Visible = True
    xlBook = xlApp.Workbooks.Open(File_path)

    # make Excel visible (1 is True, 0 is False)
    xlApp.Visible = 0

    # makes a reference to the RawData page
    xlSheet = xlBook.Sheets('User')

    # Grabbing the data and assigning it a nae
    WaferName = xlSheet.Range('A6').Value.encode('utf8')
    Thickness = float(xlSheet.Range('B6').Value)
    Resisitivity = float(xlSheet.Range('C6').Value)
    Doping = float(xlSheet.Range('J9').Value)
    SampleType = xlSheet.Range('D6').Value.encode('utf8')
    AnalysisMode = xlSheet.Range('H6').Value.encode('utf8')
    OpticalConstant = float(xlSheet.Range('E6').Value)

    # makes a reference to the RawData page
    xlSheet = xlBook.Sheets('Settings')
    A = float(xlSheet.Range('C6').Value)
    B = float(xlSheet.Range('C7').Value)
    C = float(xlSheet.Range('C8').Value)

    RefCell = float(xlSheet.Range('C5').Value)

    xlBook.Close(SaveChanges=0)
    xlApp.Application.Quit()
    # Getting all those names into a dictionary
    Dic = locals()
    # Removing the names i don't want in the dictionary
    # print Dic
    for i in ['xlApp', 'xlBook', 'File_path', 'xlSheet']:
        del Dic[i]
    # print Dic
    return Dic


def ExtractRawData(File_path, Plot=False):
    '''
    This grabs the calculated data from sinton
    Extracts columns A-G of the raw data page.
    Can't extract columns H-I as it is not complete data and this would take some more work
    This outputs the data as a structured array, with the names as the column
    '''

    if sys.platform == "win32":
        data = _dispatch_Sinton2014_ExtractRawData(File_path)
    elif:
        data = _openpyxls_Sinton2014_ExtractRawData(File_path)

    if Plot:
        plt.plot(Out['Apparent CD'], Out['Tau (sec)'], '--')
        plt.loglog()

def _openpyxls_Sinton2014_ExtractRawData(File_path):
    # import the workbooke
    wb = load_workbook('python_excel_read.xlsx', read_only=True)

    assert xlBook.Sheets('Calc')
    xlSheet = xlBook.Sheets('Calc')
    Values = np.asarray(xlSheet.Range("A9:I133").Value, dtype=np.float64)

    headers = tuple([[j.encode('utf8') for j in i]
                     for i in xlSheet.Range('A8:I8').Value][0])

    Values2 = np.asarray(xlSheet.Range("O9:P133").Value, dtype=np.float64)
    headers2 = tuple([[j.encode('utf8') for j in i]
                      for i in xlSheet.Range('O8:P8').Value][0])

    # print headers
    # makes a reference to the RawData page
    # xlSheet = xlBook.Sheets('RawData')
    # print Values.dtype.names
    # Get the values from the page
    # Values = np.asarray(xlSheet.Range("A2:G126").Value)
    # headers = tuple([[j.encode('utf8') for j in i]
    #                  for i in xlSheet.Range('A1:G1').Value][0])

    # print type(headers),len(headers)

    xlBook.Close(SaveChanges=0)
    xlApp.Application.Quit()

    Values = np.hstack((Values, Values2))
    headers += headers2

    Out = Values.view(dtype=zip(headers, ['float64'] * len(headers))).copy()

    # Too see it working
    # print Out.dtype.names

    return Out


def _dispatch_Sinton2014_ExtractRawData(File_path):
    # import the Dispatch library, get a reference to an Excel instance
    xlApp = Dispatch("Excel.Application")
    # Set the Excel file name, working directory and path
    # This is encase you don't want to use the directory you are in
    # File_path = os.path.join(working_dir, File)

    # opens the excel book
    xlBook = xlApp.Workbooks.Open(File_path)

    # make Excel visible (1 is True, 0 is False)
    xlApp.Visible = 0

    xlSheet = xlBook.Sheets('Calc')
    Values = np.asarray(xlSheet.Range("A9:I133").Value, dtype=np.float64)

    headers = tuple([[j.encode('utf8') for j in i]
                     for i in xlSheet.Range('A8:I8').Value][0])

    Values2 = np.asarray(xlSheet.Range("O9:P133").Value, dtype=np.float64)
    headers2 = tuple([[j.encode('utf8') for j in i]
                      for i in xlSheet.Range('O8:P8').Value][0])

    # print headers
    # makes a reference to the RawData page
    # xlSheet = xlBook.Sheets('RawData')
    # print Values.dtype.names
    # Get the values from the page
    # Values = np.asarray(xlSheet.Range("A2:G126").Value)
    # headers = tuple([[j.encode('utf8') for j in i]
    #                  for i in xlSheet.Range('A1:G1').Value][0])

    # print type(headers),len(headers)

    xlBook.Close(SaveChanges=0)
    xlApp.Application.Quit()

    Values = np.hstack((Values, Values2))
    headers += headers2

    Out = Values.view(dtype=zip(headers, ['float64'] * len(headers))).copy()

    # Too see it working
    # print Out.dtype.names

    return Out
