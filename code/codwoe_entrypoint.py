import defmod, revdict, check_output, score

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="demo script for participants")
    subparsers = parser.add_subparsers(dest="command", required=True)
    parser_defmod = defmod.get_parser(parser=subparsers.add_parser("defmod", help="run a definition modeling baseline"))
    parser_revdict = revdict.get_parser(parser=subparsers.add_parser("revdict", help="run a reverse dictionary baseline"))
    parser_check_output = check_output.get_parser(parser=subparsers.add_parser("check-format", help="check the format of a submission file"))
    parser_score = score.get_parser(parser=subparsers.add_parser("score", help="evaluate a submission"))
    args = parser.parse_args()
    if args.command == "defmod":
        defmod.main(args)
    elif args.command == "revdict":
        revdict.main(args)
    elif args.command == "check-format":
        check_output.main(args.submission_file)
    elif args.command == "score":
        score.main(args)
