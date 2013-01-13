class Logger:
    def __init__(self,filename):
        self.logfile = open("{filename}.log".format(filename=filename), 'a')
        self.logfile.write('==== Starting log ====\n')
    def __del__(self):
        self.logfile.close()
    def log(self,message):
        self.logfile.write("{message}\n".format(message=message))
