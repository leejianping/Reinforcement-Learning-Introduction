V_fb = 0;
V_pub =0;
V_c1 = 0;
V_c2 = 0;
V_c3 = 0;
V_pass = 0;
V_slp = 0;
gamma =0.9;

for n in  range(1,3):
    V_fb_next = 0.9*(-1+gamma*V_fb) + 0.1*(-1+gamma*V_c1)
    V_pub_next = 0.2*(1+gamma*V_c1) + 0.4*(1+gamma*V_c2)+0.4*(1+gamma*V_c3)
    V_c1_next = 0.5*(-2+gamma*V_c2) + 0.5*(-2+gamma*V_fb)
    V_c2_next = 0.2*(-2+gamma*V_slp) + 0.8*(-2+gamma*V_c3)
    V_c3_next = 0.6*(-2+gamma*V_pass) + 0.4*(-2+gamma*V_pub)
    v_pass_next = 1*(10 + 0.9*V_slp)

    V_fb = V_fb_next;
    V_pub = V_pub_next;
    V_c1 = V_c1_next;
    V_c2 = V_c2_next;
    V_c3 = V_c3_next;
    v_pass =v_pass_next



print("fb = ",V_fb);
print("pub=",V_pub);
print("V_c1 =",V_c1)
print("V_c2 =",V_c2)
print("V_c3 =",V_c3)
