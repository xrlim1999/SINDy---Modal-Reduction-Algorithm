def time_integration(x, xdot, t, poly, regrlist, ind, method="RK45"):
    """
    Integrate dxdt --> returns x(t)

    Input: xdot --> array containing derivative terms to be integrated
           t    --> array containing corresponding time to each derivative
    
    Output: x_es     --> integrated x terms
            xdot_est --> predicted xdot terms at each x{k} step
    """

    # number of "x" terms expected
    n_targets = len(xdot[0,:])

    # array of estimated results to be returned by function
    x_est = x.copy()
    xdot_est = xdot.copy()

    # Forward Euler (1st-oder)
    if method == "FE":
        """ parameters """
        xcurrent = x[0,:].copy().reshape(1,-1)  # x{k} terms used to predict xdot{k}
        f1 = x[0,:].copy()                      # array to store xdot{k} terms

        for k in range(len(xdot[:,0]) - 1):
            # time-step
            dt = t[k+1] - t[k]

            """ predict f1 at current step x{k} """
            # copy current state x{k}
            xcurrent = x_est[k,:].reshape(1,-1)

            # transform data into polynomial
            xcurrent_poly = poly.fit_transform(xcurrent)

            # predict f1
            for i in range(len(xdot[0,:])):
                xdot_est[k,i] = regrlist[i].predict(xcurrent_poly[:, ind[i]])

                """ take full FW step using f1 """
                x_est[k+1,i] = x_est[k,i] + xdot_est[k,i] * dt
        

    # Runge-Kutta 2nd-order time-integration
    elif method =="RK2":
        """ parameters """
        xcurrent = x[0,:].copy().reshape(1,-1)  # x{k} terms used to predict xdot{k}
        xhalf = x[0,:].copy().reshape(1,-1)     # x terms at half FE step
        f1 = x[0,:].copy()                      # array to store xdot{k} terms at x{k}

        for k in range(len(xdot[:,0]) - 1):
            # time-step
            dt = t[k+1] - t[k]

            """ predict f1 at current step x{k} """
            # copy current state x{k}
            xcurrent = x_est[k,:].reshape(1,-1)

            # transform data into polynomial
            xcurrent_poly = poly.fit_transform(xcurrent)

            # predict f1
            for i in range(len(xdot[0,:])):
                f1[i] = regrlist[i].predict(xcurrent_poly[:, ind[i]])

                """ take half FW step using f1 """
                xhalf[0,i] = x_est[k,i] + f1[i] * dt
            
            """ predict f2 at half step x{k+dt/2} """
            # transform x{half-step} data into polynomial
            xhalf_poly = poly.fit_transform(xhalf)

            # predict f2 at half FE step
            for i in range(n_targets):
                xdot_est[k,i] = regrlist[i].predict(xhalf_poly[:, ind[i]])

                """ compute x{k+1} using f2 """
                x_est[k+1,i] = x_est[k,i] + xdot_est[k,i] * dt


    # Runge-Kutta 4th-order time-integration
    elif method =="RK45":
        """ parameters """
        xK = x[0,:].copy().reshape(1,-1)    # x{k} terms used to predict xdot{k}

        # arrays to store x terms at intermediate steps
        x2 = x[0,:].copy().reshape(1,-1)
        x3 = x2
        x4 = x3

        # arrays to store xdot{k} terms at x{k}
        f1 = x[0,:].copy()
        f2 = f1
        f3 = f1
        f4 = f1

        for k in range(len(xdot[:,0]) - 1):
            # time-step
            dt = t[k+1] - t[k]

            """ predict f1 at current step x{k} """
            # copy current state x{k}
            xK = x_est[k,:].reshape(1,-1)

            # transform data into polynomial
            xK_poly = poly.fit_transform(xK)

            # predict f1
            for i in range(n_targets):
                f1[i] = regrlist[i].predict(xK_poly[:, ind[i]])

                """ take half FW step using f1 """
                x2[0,i] = x_est[k,i] + f1[i] * dt/2


            """ predict f2 at half step x{k+dt/2} using f1 """
            # transform half-step data into polynomial
            x2_poly = poly.fit_transform(x2)

            # predict f2
            for i in range(n_targets):
                f2[i] = regrlist[i].predict(x2_poly[:, ind[i]])

                """ take half FW step using f2 """
                x3[0,i] = x_est[k,i] + f2[i] * dt/2
            

            """ predict f3 at half step x{k+dt/2} using f2 """
            # transform half-step data into polynomial
            x3_poly = poly.fit_transform(x3)

            for i in range(n_targets):
                f3[i] = regrlist[i].predict(x3_poly[:, ind[i]])

                """ take half FW step using f2 """
                x3[0,i] = x_est[k,i] + f3[i] * dt/2


            """ predict f4 at full step x{k+dt} using f3 """
            # transform half-step data into polynomial
            x4_poly = poly.fit_transform(x4)

            for i in range(n_targets):
                f4[i] = regrlist[i].predict(x4_poly[:, ind[i]])

                """ iteration x{k} """
                ffinal = (f1[i] + 2*f2[i] + 2*f3[i] + f4[i]) / 6
                x_est[k+1,i] = x_est[k,i] + ffinal * dt
                xdot_est[k,i] = ffinal

    return x_est, xdot_est