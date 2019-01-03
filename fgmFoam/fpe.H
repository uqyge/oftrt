#include <fenv.h>

class FPExceptionsGuard
{
    static const int s_FPGuardExMask = FE_ALL_EXCEPT;

  public:
    FPExceptionsGuard()
    {
        feclearexcept(s_FPGuardExMask);

        int cw = fegetexcept();
        m_ControlWord = cw;
        cw |= s_FPGuardExMask;

        fedisableexcept(cw);
    }

    ~FPExceptionsGuard()
    {
        feclearexcept(s_FPGuardExMask);
        feenableexcept(m_ControlWord);
    }

  private:
    int m_ControlWord;
};