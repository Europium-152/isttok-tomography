import numpy as np
from skimage.draw import ellipse

def mfi(signals,projections,stop_criteria,alpha_1,alpha_2,alpha_3,max_iterations,guess=None):
    
    P = projections.reshape((projections.shape[0], -1))
    n_rows = projections.shape[1]
    n_cols = projections.shape[2]   
        
    # x and y grandient matrices ----------------------------------------------

    Dh = np.eye(n_rows*n_cols) - np.roll(np.eye(n_rows*n_cols), 1, axis=1)
    Dv = np.eye(n_rows*n_cols) - np.roll(np.eye(n_rows*n_cols), n_cols, axis=1)
    
    print('Dh:', Dh.shape, Dh.dtype)
    print('Dv:', Dv.shape, Dv.dtype)
    
    # norm matrix --------------------------------------------------------------
    
    ii, jj = ellipse(n_rows//2, n_cols//2, n_rows//2, n_cols//2)
    mask = np.ones((n_rows, n_cols))
    mask[ii,jj] = 0.
    
    Io = np.eye(n_rows*n_cols) * mask.flatten()
    
    print('Io:', Io.shape, Io.dtype)
    
    
    # Regularization parameters ------------------------------------------------
    
    alpha_1 = 1e-5
    alpha_2 = alpha_1
    alpha_3 = alpha_1*10
    
    # p transpose and PtP ------------------------------------------------------
    Pt = np.transpose(P)
    PtP = np.dot(Pt, P)
    
    # Norm matrix transposed ---------------------------------------------------
    ItIo = np.dot(np.transpose(Io), Io)
    
    
    # Descriminate between single and multiple reconstructions mode --------------
    _signals=np.array(signals)
    if len(_signals.shape)==1:
        f_list=[_signals]
    elif len(_signals.shape)==2:
        f_list=_signals
    else:
        raise ValueError("signals must be 1 dim for single reconstruction or 2 dim for multiple reconstrution")    
    
    ######################  FIRST ITERATION   ##################################
    
    if guess==None:
        # Weight matrix, first iteration sets W to 1 -------------------------------
        W=np.eye(n_rows*n_cols)
        
        # Fisher information (weighted derivatives) --------------------------------
        DtWDh=np.dot(np.transpose(Dh), np.dot(W, Dh))
        DtWDv=np.dot(np.transpose(Dv), np.dot(W, Dv))
        
        # Inversion and calculation of vector g, storage of first guess ------------
        inv = np.linalg.inv(PtP + alpha_1*DtWDh + alpha_2*DtWDv + alpha_3*ItIo)
        M = np.dot(inv, Pt)
        g_old = np.dot(M, f_list[0])
        first_g = np.array(g_old)
    else: # TODO: make sure guess is compatible before implementing!
        print ('Implementation Error: giving an initial guess is not implemented yet, too bad')
        return 0
        g_old = np.array(guess)
        first_g = guess
        
    g_list=[]    
        
    # Iterative process --------------------------------------------------------
    for f in f_list:
        i=0
        while True:
            
            i=i+1;
            
            W=np.diag(1.0/np.abs(g_old))
            
            DtWDh=np.dot(np.transpose(Dh), np.dot(W, Dh))
            DtWDv=np.dot(np.transpose(Dv), np.dot(W, Dv))
            
            inv = np.linalg.inv(PtP + alpha_1*DtWDh + alpha_2*DtWDv + alpha_3*ItIo)
            M = np.dot(inv, Pt)
            g_new = np.dot(M, f)
            
            error=np.sum(np.abs(g_new-g_old))/len(g_new)
            
            print (error)
            
            g_old=np.array(g_new) # Explicitly copy because python will not
                                  # TODO: Swaping instead of copying
                                  
            if error<stop_criteria:
                print ("Minimum Fisher converged after ",i," iterations.")
                break
            
            if i>max_iterations:
                print ("WARNING: Minimum Fisher did not converge after ",i," iterations.")
                break
            
            
        g_list.append(g_new)
        
    # Return correctly either a single or multiple reconstructions -------------
    if len(_signals.shape)==1:
        return (g_list[0],first_g)
    
    elif len(_signals.shape)==2:
        return (g_list,first_g)        
    
    
    
       