The multiprocessing seems to be broken with the old version of the files since Python 3.8 producing a typical error :

RuntimeError: 
        An attempt has been made to start a new process before the
        current process has finished its bootstrapping phase.

        This probably means that you are not using fork to start your
        child processes and you have forgotten to use the proper idiom
        in the main module:

            if __name__ == '__main__':
                freeze_support()
                ...

        The "freeze_support()" line can be omitted if the program
        is not going to be frozen to produce an executable.

The present 5 documents should resolve that issue. 

Deep aknowledgement to Romain Allart for managing to solve the issue. 