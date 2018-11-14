// #include "listOptions.H"
// bool success = sample.infer(b);
Foam::argList args(argc, argv);
// bool success = sample.infer(b);
if (!args.checkRootCase())
{
    Foam::FatalError.exit();
}
// bool success = sample.infer(b);
// #include "listOutput.H"